"""NCU Output Parser and Bottleneck Diagnoser.

Extracts target metrics from NVIDIA Nsight Compute CLI (ncu) CSV or dict exports
and constructs a standardized PIRReport with rule-based bottleneck diagnostics.
"""

from __future__ import annotations
import csv
import io
from typing import Dict, Any, List, Optional
from gpuperf.schema.pir import (
    PIRReport,
    HardwareInfo,
    LaunchConfig,
    PerformanceMetrics,
    WarpStallReason,
    MemoryEfficiency,
    BottleneckDiagnosis
)


class NCUParser:
    """Parses NCU metrics and generates structured PIR reports."""

    # Key metric mappings from NCU CLI metric names
    METRIC_MAP = {
        "duration_ns": ["gpu__time_duration.sum", "Duration"],
        "compute_sol": ["sm__throughput.avg.pct_of_peak_sustained_elapsed", "Compute (SM) Throughput"],
        "dram_sol": ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "DRAM Throughput"],
        "achieved_occupancy": ["sm__warps_active.avg.pct_of_peak_sustained_active", "Achieved Occupancy"],
        "theoretical_occupancy": ["sm__maximum_warps_per_active_cycle_pct", "Theoretical Occupancy"],
        "regs_per_thread": ["launch__registers_per_thread", "Registers Per Thread"],
        "shared_mem_bytes": ["launch__shared_mem_per_block_allocated", "Static Shared Memory"],
        "spill_ld_bytes": ["smsp__sass_inst_executed_op_local_ld.sum"],
        "spill_st_bytes": ["smsp__sass_inst_executed_op_local_st.sum"],
        "global_load_eff": ["smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"],
        "global_store_eff": ["smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct"],
        "bank_conflicts": ["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"]
    }

    @classmethod
    def diagnose(cls, pir: PIRReport) -> BottleneckDiagnosis:
        """Run rule-based hardware diagnostic logic to establish ground truth."""
        launch = pir.launch_config
        perf = pir.performance_metrics
        mem = perf.memory_efficiency

        reasons: List[str] = []
        actions: List[str] = []

        # 1. Register Spilling
        if launch.spill_loads_bytes > 0 or launch.spill_stores_bytes > 0:
            primary = "RegisterSpill"
            explanation = (
                f"Kernel is spilling registers to local memory ({launch.spill_loads_bytes}B loads, "
                f"{launch.spill_stores_bytes}B stores). High register pressure ({launch.registers_per_thread} "
                f"regs/thread) is forcing compiler to store live registers in slow local memory/L1 cache."
            )
            actions = [
                "Reduce tile dimensions (e.g. BLOCK_M / BLOCK_N in Triton)",
                "Reduce unrolling factor or simplify loop bodies",
                "Pass compiler flag --maxrregcount or reduce num_stages in Triton"
            ]
            return BottleneckDiagnosis(primary_bottleneck=primary, confidence=0.98, explanation=explanation, suggested_actions=actions)

        # 2. Memory Bandwidth vs Compute Bound
        if perf.sol_memory_dram_pct >= 65.0 and perf.sol_compute_pct < 45.0:
            primary = "MemoryBandwidthBound"
            explanation = (
                f"Kernel is heavily memory bandwidth bound (DRAM SOL: {perf.sol_memory_dram_pct:.1f}% vs "
                f"Compute SOL: {perf.sol_compute_pct:.1f}%). Execution is limited by off-chip DRAM transfers."
            )
            actions = [
                "Increase arithmetic intensity by tiling more operations in shared memory",
                "Fuse downstream elementwise or activation operations into this kernel",
                "Check data types: downcast FP32 to BF16/FP16 or FP8 if precision allows"
            ]
            return BottleneckDiagnosis(primary_bottleneck=primary, confidence=0.92, explanation=explanation, suggested_actions=actions)

        if perf.sol_compute_pct >= 65.0:
            primary = "ComputeBound"
            explanation = (
                f"Kernel is compute bound (SM SOL: {perf.sol_compute_pct:.1f}%). SM execution units are well saturated."
            )
            actions = [
                "Utilize Tensor Core instructions (wmma / mma.sync or Triton tl.dot)",
                "Optimize instruction-level parallelism and instruction mix",
                "Ensure floating point operations use native fast-math intrinsics"
            ]
            return BottleneckDiagnosis(primary_bottleneck=primary, confidence=0.90, explanation=explanation, suggested_actions=actions)

        # 3. Low Occupancy Due to Resource Limits
        if perf.achieved_occupancy_pct < 35.0:
            if launch.registers_per_thread >= 64:
                primary = "RegisterPressureOccupancy"
                explanation = (
                    f"Occupancy is severely limited ({perf.achieved_occupancy_pct:.1f}% achieved) by excessive "
                    f"register usage ({launch.registers_per_thread} regs/thread), reducing active warps per SM."
                )
                actions = [
                    "Reduce tile size or warp count to lower live variables per thread",
                    "Reuse temporary variable buffers to reduce register lifetime"
                ]
                return BottleneckDiagnosis(primary_bottleneck=primary, confidence=0.88, explanation=explanation, suggested_actions=actions)

            if launch.shared_mem_per_block_kb >= 48.0:
                primary = "SharedMemoryOccupancy"
                explanation = (
                    f"Occupancy is constrained by shared memory allocation ({launch.shared_mem_per_block_kb:.1f} KB/block), "
                    f"limiting the number of concurrent thread blocks per SM."
                )
                actions = [
                    "Reduce shared memory buffer footprints (e.g. smaller multi-buffering stages)",
                    "Pad shared memory allocations only when necessary for bank conflict avoidance"
                ]
                return BottleneckDiagnosis(primary_bottleneck=primary, confidence=0.88, explanation=explanation, suggested_actions=actions)

        # 4. Memory Latency / Stall Long Scoreboard
        for stall in perf.warp_stall_reasons:
            if "Long Scoreboard" in stall.reason and stall.pct >= 40.0:
                primary = "GlobalMemoryLatency"
                explanation = (
                    f"Warps spend {stall.pct:.1f}% of cycles stalled on Long Scoreboard (waiting for global memory loads). "
                    f"Global memory latency is not being hidden by active warps or prefetching."
                )
                actions = [
                    "Increase software pipelining (e.g. increase num_stages in Triton from 2 to 3 or 4)",
                    "Vectorize global loads using float4 (CUDA) or 128-bit aligned block pointers (Triton)",
                    "Ensure coalesced 128-byte memory transactions across warp threads"
                ]
                return BottleneckDiagnosis(primary_bottleneck=primary, confidence=0.85, explanation=explanation, suggested_actions=actions)

        # Default
        return BottleneckDiagnosis(
            primary_bottleneck="BalancedOrLatencyBound",
            confidence=0.70,
            explanation=f"Balanced resource utilization (Compute: {perf.sol_compute_pct:.1f}%, DRAM: {perf.sol_memory_dram_pct:.1f}%).",
            suggested_actions=["Profile instruction mix and inspect loop trip counts"]
        )

    @classmethod
    def parse_csv(cls, csv_content: str, kernel_name: str, language: str = "triton", hardware: Optional[HardwareInfo] = None) -> PIRReport:
        """Parse raw NCU CSV export into a PIRReport."""
        reader = csv.DictReader(io.StringIO(csv_content))
        raw_metrics: Dict[str, float] = {}

        for row in reader:
            metric_name = row.get("Metric Name", "").strip()
            metric_val = row.get("Metric Value", "").strip().replace(",", "")
            try:
                raw_metrics[metric_name] = float(metric_val)
            except (ValueError, TypeError):
                continue

        # Extract values
        latency_ns = raw_metrics.get("gpu__time_duration.sum", 0.0)
        latency_us = latency_ns / 1000.0 if latency_ns > 0 else 0.0

        compute_sol = raw_metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
        dram_sol = raw_metrics.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
        achieved_occ = raw_metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)
        theo_occ = raw_metrics.get("sm__maximum_warps_per_active_cycle_pct", 0.0)

        regs = int(raw_metrics.get("launch__registers_per_thread", 32))
        shmem_bytes = raw_metrics.get("launch__shared_mem_per_block_allocated", 0.0)
        spill_ld = int(raw_metrics.get("smsp__sass_inst_executed_op_local_ld.sum", 0))
        spill_st = int(raw_metrics.get("smsp__sass_inst_executed_op_local_st.sum", 0))

        load_eff = raw_metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct", 100.0)
        store_eff = raw_metrics.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct", 100.0)
        conflicts = int(raw_metrics.get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 0))

        # Stalls
        stalls = []
        for name, pct in raw_metrics.items():
            if "warp_issue_stalled" in name and pct > 5.0:
                short_name = name.split("stalled_")[-1].replace("_per_warp_active.pct", "").replace("_", " ").title()
                stalls.append(WarpStallReason(reason=short_name, pct=pct))
        stalls.sort(key=lambda s: s.pct, reverse=True)

        hw = hardware or HardwareInfo()
        launch = LaunchConfig(
            registers_per_thread=regs,
            shared_mem_per_block_kb=shmem_bytes / 1024.0,
            spill_loads_bytes=spill_ld,
            spill_stores_bytes=spill_st
        )
        perf = PerformanceMetrics(
            latency_us=latency_us,
            sol_compute_pct=compute_sol,
            sol_memory_dram_pct=dram_sol,
            achieved_occupancy_pct=achieved_occ,
            theoretical_occupancy_pct=theo_occ,
            warp_stall_reasons=stalls,
            memory_efficiency=MemoryEfficiency(
                global_load_efficiency_pct=load_eff,
                global_store_efficiency_pct=store_eff,
                shared_memory_bank_conflicts=conflicts
            )
        )

        pir = PIRReport(
            kernel_name=kernel_name,
            language=language,
            hardware=hw,
            launch_config=launch,
            performance_metrics=perf
        )
        pir.diagnosis = cls.diagnose(pir)
        return pir

