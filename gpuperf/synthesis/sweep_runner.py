"""Automated Parametric Sweep Runner for GPU Kernels.

Designed to run on NVIDIA L4 (or compatible GPU) to systematically explore
the optimization landscape (tile sizes, warp counts, software pipeline stages)
and collect real execution times and NCU profiles for positive & negative trajectory generation.
"""

from __future__ import annotations
import itertools
from typing import List, Dict, Any, Callable


class ParameterSweepRunner:
    """Explores kernel configurations and identifies Pareto-optimal vs regressed variants."""

    # Default grid space for Triton GEMM / Attention operators
    DEFAULT_SWEEP_CONFIG = {
        "BLOCK_M": [32, 64, 128],
        "BLOCK_N": [32, 64, 128],
        "BLOCK_K": [32, 64],
        "num_warps": [2, 4, 8],
        "num_stages": [2, 3, 4]
    }

    @classmethod
    def generate_configurations(cls, config_space: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
        """Generate all combinations from parameter space."""
        space = config_space or cls.DEFAULT_SWEEP_CONFIG
        keys = list(space.keys())
        values = list(space.values())
        
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs

    @classmethod
    def find_chosen_and_rejected(
        cls,
        results: List[Dict[str, Any]],
        baseline_time_us: float
    ) -> Dict[str, Any]:
        """Classify evaluated variants into fast winners (chosen) and regressed/spilled variants (rejected)."""
        valid_results = [r for r in results if r.get("is_valid", False) and r.get("latency_us", float("inf")) > 0]
        if not valid_results:
            return {}

        # Sort by latency ascending (fastest first)
        valid_results.sort(key=lambda x: x["latency_us"])

        fastest = valid_results[0]
        slowest = valid_results[-1]

        best_speedup = baseline_time_us / fastest["latency_us"] if fastest["latency_us"] > 0 else 1.0
        worst_speedup = baseline_time_us / slowest["latency_us"] if slowest["latency_us"] > 0 else 1.0

        return {
            "baseline_latency_us": baseline_time_us,
            "best_variant": fastest,
            "best_speedup": round(best_speedup, 3),
            "worst_variant": slowest,
            "worst_speedup": round(worst_speedup, 3),
            "all_evaluated_count": len(results)
        }

