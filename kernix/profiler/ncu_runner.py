"""NCU Execution Runner.

Executes a target CUDA binary or Python Triton script under NVIDIA Nsight Compute CLI (ncu)
and collects metrics into CSV or JSON format. Includes a mock execution mode for local testing.
"""

from __future__ import annotations
import subprocess
import os
import shutil
import tempfile
from typing import List, Optional, Dict
from kernix.schema.pir import PIRReport, HardwareInfo
from kernix.profiler.ncu_parser import NCUParser


class NCURunner:
    """Wrapper around NVIDIA Nsight Compute CLI."""

    DEFAULT_METRICS = [
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__maximum_warps_per_active_cycle_pct",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_allocated",
        "smsp__sass_inst_executed_op_local_ld.sum",
        "smsp__sass_inst_executed_op_local_st.sum",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct"
    ]

    def __init__(self, ncu_path: Optional[str] = None):
        self.ncu_path = ncu_path or shutil.which("ncu") or "ncu"

    def is_available(self) -> bool:
        """Check if ncu CLI is installed and accessible."""
        return shutil.which(self.ncu_path) is not None

    def profile_command(
        self,
        command: List[str],
        kernel_name: str,
        language: str = "triton",
        metrics: Optional[List[str]] = None,
        timeout_sec: int = 120
    ) -> PIRReport:
        """Execute command under ncu and parse the metrics."""
        if not self.is_available():
            raise RuntimeError(f"NCU CLI '{self.ncu_path}' not found on system. Use mock mode or install NVIDIA Nsight Compute.")

        metric_list = metrics or self.DEFAULT_METRICS
        metric_arg = ",".join(metric_list)

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as tmp_csv:
            tmp_path = tmp_csv.name

        try:
            ncu_cmd = [
                self.ncu_path,
                "--csv",
                f"--page=raw",
                f"--metrics={metric_arg}",
                f"--log-file={tmp_path}",
                "--target-processes=all"
            ] + command

            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )

            if result.returncode != 0:
                raise RuntimeError(f"NCU profiling failed with code {result.returncode}:\n{result.stderr}")

            with open(tmp_path, "r", encoding="utf-8") as f:
                csv_data = f.read()

            return NCUParser.parse_csv(csv_data, kernel_name=kernel_name, language=language)

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def mock_profile(
        kernel_name: str,
        language: str = "triton",
        scenario: str = "register_spill"
    ) -> PIRReport:
        """Generate a realistic mock PIR report for testing pipelines without a GPU."""
        hw = HardwareInfo(device="NVIDIA L4", arch="sm_89", num_sms=58)
        
        if scenario == "register_spill":
            csv_sample = """Metric Name,Metric Value
gpu__time_duration.sum,312000
sm__throughput.avg.pct_of_peak_sustained_elapsed,22.4
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,35.1
sm__warps_active.avg.pct_of_peak_sustained_active,18.5
sm__maximum_warps_per_active_cycle_pct,25.0
launch__registers_per_thread,124
launch__shared_mem_per_block_allocated,16384
smsp__sass_inst_executed_op_local_ld.sum,4096
smsp__sass_inst_executed_op_local_st.sum,4096
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,85.0
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,90.0
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,0
smsp__warp_issue_stalled_wait_per_warp_active.pct,38.2
"""
        elif scenario == "memory_bound":
            csv_sample = """Metric Name,Metric Value
gpu__time_duration.sum,145000
sm__throughput.avg.pct_of_peak_sustained_elapsed,18.2
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,84.6
sm__warps_active.avg.pct_of_peak_sustained_active,42.0
sm__maximum_warps_per_active_cycle_pct,50.0
launch__registers_per_thread,32
launch__shared_mem_per_block_allocated,8192
smsp__sass_inst_executed_op_local_ld.sum,0
smsp__sass_inst_executed_op_local_st.sum,0
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,55.0
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,60.0
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,0
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,48.5
"""
        else:  # optimized
            csv_sample = """Metric Name,Metric Value
gpu__time_duration.sum,82000
sm__throughput.avg.pct_of_peak_sustained_elapsed,74.5
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,48.2
sm__warps_active.avg.pct_of_peak_sustained_active,65.0
sm__maximum_warps_per_active_cycle_pct,75.0
launch__registers_per_thread,40
launch__shared_mem_per_block_allocated,32768
smsp__sass_inst_executed_op_local_ld.sum,0
smsp__sass_inst_executed_op_local_st.sum,0
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,95.0
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,95.0
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,0
smsp__warp_issue_stalled_wait_per_warp_active.pct,12.0
"""
        return NCUParser.parse_csv(csv_sample, kernel_name=kernel_name, language=language, hardware=hw)

