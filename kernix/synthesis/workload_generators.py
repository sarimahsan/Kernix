"""Parametric Workload Generators for GPU Profiling Sweeps.

Provides executable Triton kernels across 6 canonical GPU workload families:
1. Dense Matrix Multiplication (GEMM)
2. Fused Softmax / Attention
3. Row-wise Reductions (RMSNorm / LayerNorm)
4. Elementwise Activations & Projections (SwiGLU)
5. Quantized Dequant GEMM (INT8/FP8)
6. Masked / Irregular Reductions
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple


class WorkloadTemplates:
    """Parametric kernel source templates with config hooks."""

    # 1. Dense GEMM
    GEMM = """import torch
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
"""

    # 2. Fused Attention / Softmax
    FUSED_ATTENTION = """import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    q_ptrs = q_ptr + off_hz * stride_qh + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = k_ptr + off_hz * stride_kh + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = v_ptr + off_hz * stride_vh + (offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k)) * sm_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs + start_n * stride_vk, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_ij

    out = acc / l_i[:, None]
    out_ptrs = out_ptr + off_hz * stride_oh + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_on)
    tl.store(out_ptrs, out.to(tl.float16), mask=offs_m[:, None] < N_CTX)
"""

    # 3. RMSNorm / LayerNorm
    RMSNORM = """import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    row_ptr = x_ptr + row_idx * stride_x_row + col_offsets
    x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)
    
    variance = tl.sum(x * x, axis=0) / N
    rsqrt = tl.math.rsqrt(variance + eps)

    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    out = (x * rsqrt * w).to(tl.float16)

    out_ptr = y_ptr + row_idx * stride_y_row + col_offsets
    tl.store(out_ptr, out, mask=mask)
"""

    # 4. SwiGLU Fused Activation
    SWIGLU = """import torch
import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, out_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    g = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU(g) * u = g * sigmoid(g) * u
    silu_g = g * (1.0 / (1.0 + tl.exp(-g)))
    result = silu_g * u

    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)
"""

    # 5. INT8 Quantized GEMM with Fused Scaling
    INT8_GEMM = """import torch
import triton
import triton.language as tl

@triton.jit
def int8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_a_ptr, scales_b_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale_a = tl.load(scales_a_ptr + offs_am)[:, None]
    scale_b = tl.load(scales_b_ptr + offs_bn)[None, :]
    out = (acc.to(tl.float32) * scale_a * scale_b).to(tl.float16)

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, out, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))
"""


def get_sweep_configs(operator_name: str) -> List[Dict[str, Any]]:
    """Return the parameter search grid tailored for an operator."""
    if operator_name == "gemm":
        configs = []
        for bm in [32, 64, 128, 256]:
            for bn in [32, 64, 128, 256]:
                for bk in [32, 64]:
                    for warps in [2, 4, 8, 16]:
                        for stages in [2, 3, 4]:
                            configs.append({
                                "BLOCK_M": bm,
                                "BLOCK_N": bn,
                                "BLOCK_K": bk,
                                "GROUP_M": 8,
                                "num_warps": warps,
                                "num_stages": stages
                            })
        return configs

    elif operator_name == "fused_attention":
        configs = []
        for bm in [32, 64, 128]:
            for bn in [32, 64, 128]:
                for warps in [4, 8]:
                    for stages in [2, 3]:
                        configs.append({
                            "BLOCK_M": bm,
                            "BLOCK_N": bn,
                            "num_warps": warps,
                            "num_stages": stages
                        })
        return configs

    elif operator_name in ["rmsnorm", "swiglu"]:
        configs = []
        for bsize in [256, 512, 1024, 2048, 4096, 8192]:
            for warps in [2, 4, 8, 16]:
                configs.append({
                    "BLOCK_SIZE": bsize,
                    "num_warps": warps
                })
        return configs

    elif operator_name == "int8_gemm":
        configs = []
        for bm in [64, 128, 256]:
            for bn in [64, 128, 256]:
                for bk in [64, 128]:
                    for warps in [4, 8]:
                        configs.append({
                            "BLOCK_M": bm,
                            "BLOCK_N": bn,
                            "BLOCK_K": bk,
                            "num_warps": warps,
                            "num_stages": 3
                        })
        return configs

    return []

