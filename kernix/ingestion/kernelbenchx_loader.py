"""Loader and Normalizer for KernelBenchX (BonnieWang/KernelBenchX).

Parses tasks across 15 categories (Math, Fusion, Attention, Quantization, etc.),
paired with evaluation metrics (buildability, correctness, runtime, speedup).
"""

from __future__ import annotations
import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple


class KernelBenchXLoader:
    """Ingests and converts KernelBenchX samples into standardized GPUPerf trajectories."""

    @staticmethod
    def extract_code_block(raw_text: str, default_lang: str = "triton") -> str:
        """Strip markdown fences (```python ... ``` or ```triton ... ```) and trailing docstrings."""
        if not raw_text:
            return ""
        
        # Check for triple backticks
        match = re.search(r"```(?:python|triton|cuda|cpp)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return raw_text.strip()

    @classmethod
    def parse_task_record(
        cls,
        task_record: Dict[str, Any],
        metrics_record: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse a task with baseline and generated iterations into a trajectory."""
        task_id = task_record.get("task_id") or task_record.get("file", "unknown_task")
        category = task_record.get("category")
        if not category and "/" in task_id:
            category = task_id.split("/")[0]
        category = category or "general"

        language = "triton"
        if "cuda" in task_id.lower() or ".cu" in task_id.lower():
            language = "cuda"

        baseline_code = cls.extract_code_block(
            task_record.get("baseline_code") or 
            task_record.get("reference_code") or 
            task_record.get("torch_code") or ""
        )

        # Look for predicted/optimized kernel
        optimized_code = cls.extract_code_block(
            task_record.get("predict") or 
            task_record.get("optimized_code") or 
            task_record.get("round_final_code") or ""
        )

        if not baseline_code or not optimized_code:
            return None

        # Extract execution metrics
        metrics = metrics_record or task_record.get("metrics", {})
        
        # Metrics: Call (buildable), Exe (correct), Perf (speedup)
        is_buildable = metrics.get("call", task_record.get("is_buildable", True))
        is_correct = metrics.get("exe", task_record.get("is_correct", True))
        
        # Speedup relative to baseline
        speedup = 1.0
        perf_data = metrics.get("perf", {})
        if isinstance(perf_data, dict):
            speedup = float(perf_data.get("speedup", 1.0))
            runtime_ms = float(perf_data.get("runtime_ms", 0.0))
            baseline_ms = float(perf_data.get("baseline_ms", 0.0))
            tflops = float(perf_data.get("tflops", 0.0))
            bandwidth_gbs = float(perf_data.get("bandwidth_gbs", 0.0))
        else:
            speedup = float(task_record.get("speedup", 1.0))
            runtime_ms = float(task_record.get("runtime_ms", 0.0))
            baseline_ms = float(task_record.get("baseline_ms", 0.0))
            tflops = 0.0
            bandwidth_gbs = 0.0

        return {
            "source": "BonnieWang/KernelBenchX",
            "task_id": task_id,
            "category": category,
            "language": language,
            "baseline_code": baseline_code,
            "optimized_code": optimized_code,
            "is_buildable": bool(is_buildable),
            "is_correct": bool(is_correct),
            "speedup": round(speedup, 3),
            "runtime_ms": runtime_ms,
            "baseline_ms": baseline_ms,
            "tflops": tflops,
            "bandwidth_gbs": bandwidth_gbs,
            "is_chosen": is_correct and speedup > 1.15,
            "is_rejected": (not is_correct) or (speedup < 0.95),
            "critique": task_record.get("critique") or task_record.get("feedback") or ""
        }

    @classmethod
    def process_corpus(cls, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split corpus into chosen (speedup >= 1.15x) and rejected (slowdown/broken) pairs."""
        chosen_pairs = []
        rejected_pairs = []

        for r in records:
            parsed = cls.parse_task_record(r)
            if not parsed:
                continue

            if parsed["is_chosen"]:
                chosen_pairs.append(parsed)
            elif parsed["is_rejected"]:
                rejected_pairs.append(parsed)

        return chosen_pairs, rejected_pairs
