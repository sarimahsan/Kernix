"""Trajectory Builder for SFT and DPO Dataset Formats.

Converts (Kernel, PIR, Refactored Kernel, Speedup) records into standardized
ShareGPT / OpenAI / TRL chat-template format for supervised fine-tuning and DPO.
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
from gpuperf.schema.pir import PIRReport


SYSTEM_PROMPT = """You are GPUPerf, a specialized GPU systems optimization engineer.
Your task is to analyze GPU kernels (CUDA/Triton) alongside hardware profiler evidence (NVIDIA Nsight Compute), diagnose the root-cause bottleneck, and provide a functionally correct, empirically faster refactored kernel."""


class TrajectoryBuilder:
    """Builds formatted training prompts for SFT and DPO."""

    @staticmethod
    def build_sft_sample(
        initial_code: str,
        pir_report: PIRReport,
        refactored_code: str,
        diagnosis_reasoning: str,
        measured_speedup: float,
        language: str = "triton"
    ) -> Dict[str, Any]:
        """Construct a multi-turn SFT conversation sample."""
        user_prompt = (
            f"Here is a {language.upper()} GPU kernel and its hardware profiling report from NVIDIA Nsight Compute:\n\n"
            f"```{language}\n{initial_code.strip()}\n```\n\n"
            f"{pir_report.to_compact_prompt_str()}\n\n"
            f"Analyze the profiler metrics, identify the primary hardware bottleneck, and write an optimized kernel that improves execution latency."
        )

        assistant_response = (
            f"### Bottleneck Diagnosis\n"
            f"{diagnosis_reasoning.strip()}\n\n"
            f"### Optimization Strategy\n"
            f"- **Observed Limiter:** {pir_report.performance_metrics.occupancy_limiter} / DRAM SOL {pir_report.performance_metrics.sol_memory_dram_pct:.1f}%\n"
            f"- **Target Transform:** Applied memory access restructuring, loop unrolling adjustments, and tile/warp parameter tuning.\n\n"
            f"### Optimized {language.upper()} Kernel\n"
            f"```{language}\n{refactored_code.strip()}\n```\n\n"
            f"### Verification & Performance Result\n"
            f"- **Target Hardware:** {pir_report.hardware.device} ({pir_report.hardware.arch})\n"
            f"- **Measured Speedup:** {measured_speedup:.2f}x over initial kernel\n"
            f"- **Hardware Improvement:** Eliminated memory latency stalls and improved SM throughput."
        )

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ],
            "metadata": {
                "kernel_name": pir_report.kernel_name,
                "language": language,
                "device": pir_report.hardware.device,
                "speedup": measured_speedup,
                "primary_bottleneck": pir_report.diagnosis.primary_bottleneck if pir_report.diagnosis else "Unknown"
            }
        }

    @staticmethod
    def build_dpo_sample(
        initial_code: str,
        pir_report: PIRReport,
        chosen_code: str,
        chosen_reasoning: str,
        chosen_speedup: float,
        rejected_code: str,
        rejected_reasoning: str,
        rejected_speedup: float,
        language: str = "triton"
    ) -> Dict[str, Any]:
        """Construct a chosen/rejected preference pair for DPO/ORPO training."""
        prompt = (
            f"Here is a {language.upper()} GPU kernel and its hardware profiling report from NVIDIA Nsight Compute:\n\n"
            f"```{language}\n{initial_code.strip()}\n```\n\n"
            f"{pir_report.to_compact_prompt_str()}\n\n"
            f"Analyze the profiler metrics, identify the primary hardware bottleneck, and write an optimized kernel that improves execution latency."
        )

        chosen_text = (
            f"### Bottleneck Diagnosis\n{chosen_reasoning.strip()}\n\n"
            f"### Optimized {language.upper()} Kernel\n```{language}\n{chosen_code.strip()}\n```\n\n"
            f"### Verification\nAchieved {chosen_speedup:.2f}x speedup on {pir_report.hardware.device}."
        )

        rejected_text = (
            f"### Bottleneck Diagnosis\n{rejected_reasoning.strip()}\n\n"
            f"### Optimized {language.upper()} Kernel\n```{language}\n{rejected_code.strip()}\n```\n\n"
            f"### Verification\nResulted in {rejected_speedup:.2f}x speedup (regression/spill) on {pir_report.hardware.device}."
        )

        return {
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "metadata": {
                "kernel_name": pir_report.kernel_name,
                "chosen_speedup": chosen_speedup,
                "rejected_speedup": rejected_speedup,
                "speedup_delta": round(chosen_speedup - rejected_speedup, 3)
            }
        }

