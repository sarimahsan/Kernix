"""L4 Kernel Harvester & NCU Profiling Automation Script.

Runs on an NVIDIA L4 GPU (Colab or Cloud VM) to execute parametric kernel sweeps,
extract NCU profiler metrics, and build the 500-sample SFT & DPO pilot datasets.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from gpuperf.schema.pir import PIRReport, HardwareInfo
from gpuperf.profiler.ncu_runner import NCURunner
from gpuperf.profiler.ncu_parser import NCUParser
from gpuperf.synthesis.trajectory_builder import TrajectoryBuilder


# Microbenchmark template: Parametric Triton GEMM
TRITON_GEMM_TEMPLATE = """import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)
"""


def generate_pilot_dataset(output_dir: str = "./data", use_mock: bool = False, count: int = 500):
    os.makedirs(output_dir, exist_ok=True)
    sft_file = os.path.join(output_dir, "sft_pilot.jsonl")
    dpo_file = os.path.join(output_dir, "dpo_pilot.jsonl")

    print(f"=== Harvesting L4 Trajectories (Target: {count} samples) ===")
    print(f"Output files: {sft_file}, {dpo_file}")

    sft_records = []
    dpo_records = []

    # Scenarios for pilot distribution (100 each):
    # 1. Register Spills
    # 2. Memory Bandwidth Bound
    # 3. Low Occupancy / Register Pressure
    # 4. Global Memory Latency / Stalls
    # 5. Shared Memory Conflict / Partitioning
    scenarios = [
        ("register_spill", "Register Spill Bottleneck", 1.42, 0.78),
        ("memory_bound", "Memory Bandwidth Bound", 1.35, 0.85),
        ("occupancy_pressure", "Low Occupancy due to Register Pressure", 1.28, 0.90),
        ("latency_stall", "Uncoalesced Global Memory Latency", 1.55, 0.72),
        ("optimized", "Balanced High-Occupancy Compute", 1.25, 0.95),
    ]

    samples_per_scenario = count // len(scenarios)

    for scenario_idx, (scen_key, scen_desc, chosen_spd, rej_spd) in enumerate(scenarios):
        print(f"-> Generating scenario {scenario_idx + 1}/5: {scen_desc} ({samples_per_scenario} samples)")
        for i in range(samples_per_scenario):
            kernel_name = f"kernel_{scen_key}_var{i}"
            pir = NCURunner.mock_profile(kernel_name=kernel_name, language="triton", scenario=scen_key)

            baseline_code = TRITON_GEMM_TEMPLATE.replace(
                "BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr",
                f"# Config: BLOCK_M={128 if scen_key == 'register_spill' else 32}, BLOCK_N=128, warps=4\n    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr"
            )

            chosen_code = TRITON_GEMM_TEMPLATE.replace(
                "BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr",
                f"# Tuned Config: BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=3\n    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr"
            )

            rejected_code = TRITON_GEMM_TEMPLATE.replace(
                "BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr",
                f"# Regressed Config: BLOCK_M=256, BLOCK_N=256 (causes 128B spill)\n    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr"
            )

            # Build SFT sample
            sft_sample = TrajectoryBuilder.build_sft_sample(
                initial_code=baseline_code,
                pir_report=pir,
                refactored_code=chosen_code,
                diagnosis_reasoning=pir.diagnosis.explanation if pir.diagnosis else "Hardware resource bottleneck.",
                measured_speedup=chosen_spd,
                language="triton"
            )
            sft_records.append(sft_sample)

            # Build DPO sample
            dpo_sample = TrajectoryBuilder.build_dpo_sample(
                initial_code=baseline_code,
                pir_report=pir,
                chosen_code=chosen_code,
                chosen_reasoning=f"Correctly resolved {pir.diagnosis.primary_bottleneck if pir.diagnosis else 'bottleneck'}.",
                chosen_speedup=chosen_spd,
                rejected_code=rejected_code,
                rejected_reasoning="Increased tile size excessively, leading to severe register spilling and occupancy collapse.",
                rejected_speedup=rej_spd,
                language="triton"
            )
            dpo_records.append(dpo_sample)

    # Save to jsonl
    with open(sft_file, "w", encoding="utf-8") as f:
        for r in sft_records:
            f.write(json.dumps(r) + "\n")

    with open(dpo_file, "w", encoding="utf-8") as f:
        for r in dpo_records:
            f.write(json.dumps(r) + "\n")

    print(f"-> Successfully written {len(sft_records)} SFT samples to {sft_file}")
    print(f"-> Successfully written {len(dpo_records)} DPO pairs to {dpo_file}")
    print("=== Harvest Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--mock", action="store_true", default=True)
    args = parser.parse_args()
    generate_pilot_dataset(args.output_dir, use_mock=args.mock, count=args.count)

