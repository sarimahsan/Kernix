"""Closed-Loop Verification & Correctness Gate for GPUPerf-3B.

Implements an automated 3-tier validation harness:
1. Tier 1: Syntax & Compilation Gate (Triton JIT compilation & argument verification)
2. Tier 2: Numerical Correctness Gate (Reference comparison vs torch.matmul / torch reference)
3. Tier 3: Physical Profiler Gate (Microsecond CUDA Event benchmarking vs baseline)
"""

from __future__ import annotations
import re
import traceback
from typing import Dict, Any, Tuple, Optional
import torch


class CorrectnessGate:
    """Verifies generated kernels against PyTorch golden references and measures real physical speedup."""

    @staticmethod
    def extract_triton_code(markdown_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extracts the Python/Triton code block and parsed launch configuration from LLM markdown response."""
        code_match = re.search(r"```(?:triton|python)?\s*\n(.*?)\n```", markdown_text, re.DOTALL)
        if not code_match:
            return None, {}

        code_str = code_match.group(1).strip()

        # Parse launch config comments e.g. '# Launch Config for Tesla T4: num_warps=8, num_stages=4'
        launch_config = {"num_warps": 4, "num_stages": 3}
        warps_match = re.search(r"num_warps\s*=\s*(\d+)", code_str)
        stages_match = re.search(r"num_stages\s*=\s*(\d+)", code_str)

        if warps_match:
            launch_config["num_warps"] = int(warps_match.group(1))
        if stages_match:
            launch_config["num_stages"] = int(stages_match.group(1))

        return code_str, launch_config

    @staticmethod
    def verify_gemm_kernel(
        code_str: str,
        launch_config: Dict[str, Any],
        M: int = 2048,
        N: int = 2048,
        K: int = 2048,
        baseline_latency_us: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs the 3-Tier Gate on a generated Triton GEMM kernel.
        
        Returns:
            Dict containing pass/fail status for each tier, max numerical difference,
            measured latency, and empirical speedup.
        """
        result = {
            "tier1_compilation_passed": False,
            "tier2_numerical_passed": False,
            "tier3_speedup_passed": False,
            "max_abs_error": None,
            "measured_latency_us": None,
            "speedup_vs_baseline": None,
            "rejection_reason": None
        }

        # --- Tier 1: Syntax & Compilation Gate ---
        # Triton @jit requires inspect.getsource(), so the code must be backed by a real .py file
        import tempfile
        import importlib.util
        import os

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                temp_file = f.name
                f.write(code_str)

            spec = importlib.util.spec_from_file_location("dynamic_triton_kernel", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            gemm_fn = getattr(module, "gemm_kernel", None)
        except Exception as e:
            result["rejection_reason"] = f"Tier 1 Syntax/Import Error: {e}"
            return result
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

        if gemm_fn is None:
            result["rejection_reason"] = "Tier 1 Error: 'gemm_kernel' function not found in generated code."
            return result

        # --- Tier 2: Numerical Correctness Gate ---
        try:
            import triton
            a = torch.randn((M, K), device="cuda", dtype=torch.float16)
            b = torch.randn((K, N), device="cuda", dtype=torch.float16)
            c_triton = torch.empty((M, N), device="cuda", dtype=torch.float16)

            # PyTorch golden reference
            c_ref = torch.matmul(a, b)

            num_warps = launch_config.get("num_warps", 4)
            num_stages = launch_config.get("num_stages", 3)

            # Call the kernel with standard grid and stride arguments
            def grid_fn(META):
                return (triton.cdiv(M, META.get("BLOCK_M", 64)) * triton.cdiv(N, META.get("BLOCK_N", 64)),)

            # Warmup execution (compiles Triton JIT)
            gemm_fn[grid_fn](
                a, b, c_triton,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c_triton.stride(0), c_triton.stride(1),
                num_warps=num_warps,
                num_stages=num_stages
            )
            torch.cuda.synchronize()
            result["tier1_compilation_passed"] = True

            # Check numerical equivalence
            max_diff = (c_triton - c_ref).abs().max().item()
            result["max_abs_error"] = max_diff

            # Allow FP16 tolerance: atol=0.05 (due to FP16 accumulation rounding vs FP32 ref)
            is_correct = torch.allclose(c_triton, c_ref, atol=0.05, rtol=0.01)
            if not is_correct:
                result["rejection_reason"] = f"Tier 2 Numerical Mismatch: Max error {max_diff:.4f} exceeds FP16 tolerance."
                return result

            result["tier2_numerical_passed"] = True

        except Exception as e:
            result["rejection_reason"] = f"Execution Failure: {e}\n{traceback.format_exc()}"
            return result

        # --- Tier 3: Physical Profiler Gate ---
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            iters = 15

            start_event.record()
            for _ in range(iters):
                gemm_fn[grid_fn](
                    a, b, c_triton,
                    M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c_triton.stride(0), c_triton.stride(1),
                    num_warps=num_warps,
                    num_stages=num_stages
                )
            end_event.record()
            torch.cuda.synchronize()

            latency_ms = start_event.elapsed_time(end_event) / iters
            latency_us = latency_ms * 1000.0
            result["measured_latency_us"] = latency_us

            if baseline_latency_us:
                speedup = baseline_latency_us / latency_us
                result["speedup_vs_baseline"] = speedup
                result["tier3_speedup_passed"] = speedup >= 1.05
            else:
                result["tier3_speedup_passed"] = True

        except Exception as e:
            result["rejection_reason"] = f"Tier 3 Timing Failure: {e}"
            return result

        return result

