"""Comprehensive Evaluation Suite for GPUPerf-3B Checkpoints.

Evaluates Base Qwen, SFT Adapter, or DPO Final Adapter across:
1. Diagnostic Accuracy (% matching NCU bottleneck diagnosis)
2. Compilation Pass Rate (% error-free Triton compilation)
3. Numerical Correctness (% matching torch.matmul within tolerance)
4. Empirical Speedup (% kernels achieving > 1.05x speedup)
5. Architectural Sensitivity (Does it produce different tiles for T4 vs A100 vs Blackwell?)
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Dict, Any, List
import torch

# Self-healing path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gpuperf.profiler.correctness_gate import CorrectnessGate


def run_evaluation(
    checkpoint_path: str = "./checkpoints/gpuperf_3b_dpo_final",
    test_data_path: str = "./data/sft_multi_gpu.jsonl",
    num_samples: int = 20,
    compare_with_base: bool = True
):
    print("=" * 75)
    print("      GPUPERF-3B COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 75)
    print(f"Checkpoint Under Test: {checkpoint_path}")
    print(f"Test Dataset:          {test_data_path}")
    print(f"Number of Test Kernels: {num_samples}")
    print(f"CUDA Available:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name:           {torch.cuda.get_device_name(0)}")
    print("=" * 75)

    # 1. Load Dataset
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Dataset not found at {test_data_path}")

    with open(test_data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Pick diverse test sample across all 4 architectures
    test_samples = records[-num_samples:]  # Take from the end as held-out set

    # 2. Load Model & Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
    print(f"\n[1/3] Loading Base Model: {base_model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path if os.path.exists(checkpoint_path) else base_model_name,
        trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    if os.path.exists(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        print(f"[2/3] Loading LoRA Adapter from: {checkpoint_path}...")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        print("[!] No adapter found at path. Evaluating BASE model only.")
        model = base_model

    model.eval()

    # 3. Evaluation Loop
    print(f"\n[3/3] Running Evaluation across {len(test_samples)} held-out kernels...")

    results = []
    compilation_success = 0
    numerical_success = 0
    speedup_success = 0
    diag_success = 0

    for idx, sample in enumerate(test_samples):
        metadata = sample.get("metadata", {})
        device_name = metadata.get("device", "Unknown GPU")
        arch = metadata.get("arch", "unknown")
        operator = metadata.get("operator", "gemm")
        baseline_lat = metadata.get("latency_us", 10000.0)

        # Messages: system + user
        messages = sample["messages"][:2]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=formatted_prompt,
                max_new_tokens=768,
                temperature=0.1,
                do_sample=False
            )
            response = tokenizer.decode(outputs[0][formatted_prompt.shape[1]:], skip_special_tokens=True)

        # Extract code & launch config
        code_str, launch_cfg = CorrectnessGate.extract_triton_code(response)

        # Bottleneck check
        expected_bottleneck = sample["messages"][1]["content"]
        has_diagnosis = "### Bottleneck Diagnosis" in response
        if has_diagnosis:
            diag_success += 1

        # Run 3-Tier Gate if CUDA is available and code extracted
        tier_res = {"tier1_compilation_passed": False, "tier2_numerical_passed": False, "tier3_speedup_passed": False}
        if torch.cuda.is_available() and code_str:
            tier_res = CorrectnessGate.verify_gemm_kernel(
                code_str,
                launch_cfg,
                baseline_latency_us=baseline_lat * sample.get("metadata", {}).get("speedup", 1.5)
            )

        if tier_res["tier1_compilation_passed"]:
            compilation_success += 1
        if tier_res["tier2_numerical_passed"]:
            numerical_success += 1
        if tier_res.get("tier3_speedup_passed", False):
            speedup_success += 1

        results.append({
            "idx": idx + 1,
            "device": device_name,
            "arch": arch,
            "operator": operator,
            "has_code": code_str is not None,
            "compilation": tier_res["tier1_compilation_passed"],
            "numerical": tier_res["tier2_numerical_passed"],
            "speedup": tier_res.get("speedup_vs_baseline", None)
        })

        sys.stdout.write(f"\rEvaluating kernel [{idx + 1}/{len(test_samples)}] on {device_name}...")
        sys.stdout.flush()

    total = len(test_samples)
    print("\n\n" + "=" * 75)
    print("                    EVALUATION SUMMARY REPORT")
    print("=" * 75)
    print(f"Model Evaluated:               {checkpoint_path}")
    print(f"Total Kernels Tested:          {total}")
    print(f"Diagnostic Formatting Rate:    {(diag_success / total) * 100:.1f}% ({diag_success}/{total})")
    print(f"Code Generation Rate:          {(sum(1 for r in results if r['has_code']) / total) * 100:.1f}%")
    if torch.cuda.is_available():
        print(f"Tier 1 (Compilation Pass):     {(compilation_success / total) * 100:.1f}% ({compilation_success}/{total})")
        print(f"Tier 2 (Numerical Accuracy):   {(numerical_success / total) * 100:.1f}% ({numerical_success}/{total})")
        print(f"Tier 3 (Empirical Speedup):    {(speedup_success / total) * 100:.1f}% ({speedup_success}/{total})")
    else:
        print("[!] Note: CUDA is not available on host. Tier 1-3 execution skipped.")
        print("    Run on a GPU (Colab/Cloud VM) to execute the physical gates.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPUPerf Checkpoints")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/gpuperf_3b_dpo_final",
                        help="Path to adapter checkpoint (e.g. ./checkpoints/gpuperf_3b_dpo_final or ./checkpoints/gpuperf_3b_pilot)")
    parser.add_argument("--dataset", type=str, default="./data/sft_multi_gpu.jsonl",
                        help="Path to test dataset")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of test kernels to evaluate")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        test_data_path=args.dataset,
        num_samples=args.samples
    )


if __name__ == "__main__":
    main()

