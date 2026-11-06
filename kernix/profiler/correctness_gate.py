"""Closed-Loop Verification & Correctness Gate for GPUPerf-3B / Kernix Models.

Implements an automated 3-tier validation harness:
1. Tier 1: Syntax & Compilation Gate (Triton JIT compilation & argument verification)
2. Tier 2: Numerical Correctness Gate (Reference comparison vs torch golden references)
3. Tier 3: Physical Profiler Gate (Microsecond CUDA Event benchmarking vs baseline)

Supports both GEMM (matrix multiplication) and RMSNorm operators across NVIDIA architectures.
"""

from __future__ import annotations
import os
import re
import tempfile
import importlib.util
import traceback
from typing import Dict, Any, Tuple, Optional
import torch


class CorrectnessGate:
    """Verifies generated kernels against PyTorch golden references and measures real physical speedup."""

    @staticmethod
    def extract_triton_code(markdown_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Extracts the Python/Triton code block and parsed launch configuration from LLM markdown response."""
        code_str = None
        # 1. Closed code block
        code_match = re.search(r"```(?:triton|python)?\s*\n(.*?)\n```", markdown_text, re.DOTALL)
        if code_match:
            code_str = code_match.group(1).strip()
        else:
            # 2. Unclosed code block (if generation reached token limit before final backticks)
            code_match_unclosed = re.search(r"```(?:triton|python)?\s*\n(.*)", markdown_text, re.DOTALL)
            if code_match_unclosed:
                candidate = code_match_unclosed.group(1).strip()
                if "def gemm_kernel" in candidate or "def rmsnorm_kernel" in candidate:
                    code_str = candidate
            elif "def gemm_kernel" in markdown_text or "def rmsnorm_kernel" in markdown_text:
                # 3. Raw Python code without markdown fences
                code_str = markdown_text.strip()

        if not code_str:
            return None, {}

        # Parse launch config comments and kernel parameter constants
        launch_config = {
            "num_warps": 4,
            "num_stages": 3,
            "BLOCK_SIZE": 8192,
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8
        }
        warps_match = re.search(r"num_warps\s*[:=]\s*(\d+)", code_str)
        stages_match = re.search(r"num_stages\s*[:=]\s*(\d+)", code_str)
        block_match = re.search(r"BLOCK_SIZE.*?=\s*(\d+)", code_str)
        bm_match = re.search(r"BLOCK_M.*?=\s*(\d+)", code_str)
        bn_match = re.search(r"BLOCK_N.*?=\s*(\d+)", code_str)
        bk_match = re.search(r"BLOCK_K.*?=\s*(\d+)", code_str)
        gm_match = re.search(r"GROUP_M.*?=\s*(\d+)", code_str)

        if warps_match:
            launch_config["num_warps"] = int(warps_match.group(1))
        if stages_match:
            launch_config["num_stages"] = int(stages_match.group(1))
        if block_match:
            launch_config["BLOCK_SIZE"] = int(block_match.group(1))
        if bm_match:
            launch_config["BLOCK_M"] = int(bm_match.group(1))
        if bn_match:
            launch_config["BLOCK_N"] = int(bn_match.group(1))
        if bk_match:
            launch_config["BLOCK_K"] = int(bk_match.group(1))
        if gm_match:
            launch_config["GROUP_M"] = int(gm_match.group(1))

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
        """Runs the 3-Tier Gate on a generated Triton GEMM kernel."""
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
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                temp_file = f.name
                f.write(code_str)

            spec = importlib.util.spec_from_file_location("dynamic_triton_gemm", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            gemm_fn = getattr(module, "gemm_kernel", None)
            if gemm_fn is None:
                for candidate in ["matmul_kernel", "triton_gemm", "gemm", "kernel"]:
                    if hasattr(module, candidate):
                        gemm_fn = getattr(module, candidate)
                        break
            if gemm_fn is None:
                for attr_name in dir(module):
                    if attr_name.startswith("_") or attr_name in ["torch", "triton", "tl"]:
                        continue
                    attr = getattr(module, attr_name)
                    if callable(attr) and hasattr(attr, "__name__"):
                        gemm_fn = attr
                        break
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
            result["rejection_reason"] = "Tier 1 Error: No valid GEMM kernel function found in generated code."
            return result

        # --- Tier 2: Numerical Correctness Gate ---
        try:
            import triton
            a = torch.randn((M, K), device="cuda", dtype=torch.float16)
            b = torch.randn((K, N), device="cuda", dtype=torch.float16)
            c_triton = torch.empty((M, N), device="cuda", dtype=torch.float16)

            # PyTorch golden reference
            c_ref = torch.matmul(a, b)

            block_m = launch_config.get("BLOCK_M", 128)
            block_n = launch_config.get("BLOCK_N", 128)
            num_warps = launch_config.get("num_warps", 4)
            num_stages = min(launch_config.get("num_stages", 3), 4)

            # Hardware shared memory safety check: max 164 KB per block on sm_80
            stage_bytes = (block_m * 64 + 64 * block_n) * 2
            if stage_bytes * num_stages > 160 * 1024:
                num_stages = max(2, (150 * 1024) // max(stage_bytes, 1))

            def grid_fn(META):
                bm = META.get("BLOCK_M", block_m)
                bn = META.get("BLOCK_N", block_n)
                return (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

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

    @staticmethod
    def verify_rmsnorm_kernel(
        code_str: str,
        launch_config: Dict[str, Any],
        M: int = 4096,
        N: int = 4096,
        eps: float = 1e-5,
        baseline_latency_us: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs the 3-Tier Gate on a generated Triton RMSNorm kernel."""
        result = {
            "tier1_compilation_passed": False,
            "tier2_numerical_passed": False,
            "tier3_speedup_passed": False,
            "max_abs_error": None,
            "measured_latency_us": None,
            "speedup_vs_baseline": None,
            "rejection_reason": None
        }

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                temp_file = f.name
                f.write(code_str)

            spec = importlib.util.spec_from_file_location("dynamic_triton_rmsnorm", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            norm_fn = getattr(module, "rmsnorm_kernel", None)
            if norm_fn is None:
                for candidate in ["rms_norm_kernel", "norm_kernel", "rmsnorm", "kernel"]:
                    if hasattr(module, candidate):
                        norm_fn = getattr(module, candidate)
                        break
            if norm_fn is None:
                for attr_name in dir(module):
                    if attr_name.startswith("_") or attr_name in ["torch", "triton", "tl"]:
                        continue
                    attr = getattr(module, attr_name)
                    if callable(attr) and hasattr(attr, "__name__"):
                        norm_fn = attr
                        break
        except Exception as e:
            result["rejection_reason"] = f"Tier 1 Syntax/Import Error: {e}"
            return result
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

        if norm_fn is None:
            result["rejection_reason"] = "Tier 1 Error: No valid RMSNorm kernel function found in generated code."
            return result

        try:
            import triton
            x = torch.randn((M, N), device="cuda", dtype=torch.float16)
            w = torch.ones((N,), device="cuda", dtype=torch.float16)
            y_triton = torch.zeros_like(x)

            # PyTorch golden reference for RMSNorm
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            y_ref = (x.to(torch.float32) * torch.rsqrt(variance + eps) * w.to(torch.float32)).to(torch.float16)

            num_warps = launch_config.get("num_warps", 4)
            # Ensure BLOCK_SIZE is at least next power of 2 of N so reduction covers all columns
            parsed_block = launch_config.get("BLOCK_SIZE", 8192)
            block_size = max(parsed_block, triton.next_power_of_2(N))

            def grid_fn(META):
                return (M,)

            # Warmup
            norm_fn[grid_fn](
                x, w, y_triton,
                x.stride(0), y_triton.stride(0),
                N, eps,
                num_warps=num_warps,
                BLOCK_SIZE=block_size
            )
            torch.cuda.synchronize()
            result["tier1_compilation_passed"] = True

            max_diff = (y_triton - y_ref).abs().max().item()
            result["max_abs_error"] = max_diff

            if torch.allclose(y_triton, y_ref, atol=0.08, rtol=0.02):
                result["tier2_numerical_passed"] = True
            else:
                result["rejection_reason"] = f"Tier 2 Numerical Mismatch: Max error {max_diff:.4f} exceeds FP16 tolerance."
                return result

        except Exception as e:
            result["rejection_reason"] = f"Execution Failure: {e}\n{traceback.format_exc()}"
            return result

        # Tier 3 Timing
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            iters = 25

            start_event.record()
            for _ in range(iters):
                norm_fn[grid_fn](
                    x, w, y_triton,
                    x.stride(0), y_triton.stride(0),
                    N, eps,
                    num_warps=num_warps,
                    BLOCK_SIZE=block_size
                )
            end_event.record()
            torch.cuda.synchronize()

            latency_ms = start_event.elapsed_time(end_event) / iters
            latency_us = latency_ms * 1000.0
            result["measured_latency_us"] = latency_us

            if baseline_latency_us and baseline_latency_us > 0:
                speedup = baseline_latency_us / latency_us
                result["speedup_vs_baseline"] = speedup
                result["tier3_speedup_passed"] = speedup >= 1.05
            else:
                result["tier3_speedup_passed"] = True

        except Exception as e:
            result["rejection_reason"] = f"Tier 3 Timing Failure: {e}"
            return result

        return result

    @classmethod
    def verify_kernel(
        cls,
        code_str: str,
        launch_config: Dict[str, Any],
        operator: str = "gemm",
        baseline_latency_us: Optional[float] = None
    ) -> Dict[str, Any]:
        """Automatically routes kernel to the appropriate verification gate."""
        if "rmsnorm_kernel" in code_str or operator.lower() == "rmsnorm":
            return cls.verify_rmsnorm_kernel(
                code_str,
                launch_config,
                baseline_latency_us=baseline_latency_us
            )
        else:
            return cls.verify_gemm_kernel(
                code_str,
                launch_config,
                baseline_latency_us=baseline_latency_us
            )
