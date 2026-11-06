"""Profiler Intermediate Representation (PIR) Schema.

This module defines the compact, token-efficient JSON schema representing
hardware configuration, launch parameters, NCU profiler metrics, and
bottleneck diagnoses for GPU kernels.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
from typing import List, Optional, Dict, Any


@dataclass
class HardwareInfo:
    """GPU hardware specifications."""
    device: str = "NVIDIA L4"
    arch: str = "sm_89"
    num_sms: int = 58
    max_shared_mem_per_block_kb: float = 100.0
    max_regs_per_sm: int = 65536
    peak_dram_bandwidth_gbs: float = 300.0
    peak_fp16_tflops: float = 121.0


@dataclass
class LaunchConfig:
    """Kernel launch parameters and resource allocations."""
    grid: List[int] = field(default_factory=lambda: [1, 1, 1])
    block: List[int] = field(default_factory=lambda: [128, 1, 1])
    registers_per_thread: int = 32
    shared_mem_per_block_kb: float = 0.0
    spill_loads_bytes: int = 0
    spill_stores_bytes: int = 0


@dataclass
class WarpStallReason:
    """Individual warp stall reason and percentage."""
    reason: str
    pct: float


@dataclass
class MemoryEfficiency:
    """Global and shared memory access metrics."""
    global_load_efficiency_pct: float = 100.0
    global_store_efficiency_pct: float = 100.0
    shared_memory_bank_conflicts: int = 0
    l1_cache_hit_rate_pct: float = 0.0
    l2_cache_hit_rate_pct: float = 0.0


@dataclass
class PerformanceMetrics:
    """Core Speed-of-Light (SOL), occupancy, and latency metrics."""
    latency_us: float = 0.0
    sol_compute_pct: float = 0.0
    sol_memory_dram_pct: float = 0.0
    achieved_occupancy_pct: float = 0.0
    theoretical_occupancy_pct: float = 0.0
    occupancy_limiter: str = "None"
    warp_stall_reasons: List[WarpStallReason] = field(default_factory=list)
    memory_efficiency: MemoryEfficiency = field(default_factory=MemoryEfficiency)


@dataclass
class BottleneckDiagnosis:
    """Rule-based or model-generated hardware bottleneck assessment."""
    primary_bottleneck: str = "Unknown"  # e.g., "MemoryBandwidth", "RegisterPressure", "Latency", "ComputeBound"
    confidence: float = 1.0
    explanation: str = ""
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class PIRReport:
    """Complete Profiler Intermediate Representation report for a single kernel run."""
    kernel_name: str
    language: str  # "triton" or "cuda"
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    launch_config: LaunchConfig = field(default_factory=LaunchConfig)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    diagnosis: Optional[BottleneckDiagnosis] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PIR report to Python dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize PIR report to compact JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_compact_prompt_str(self) -> str:
        """Format PIR as a token-efficient text prompt for LLM consumption (<250 tokens)."""
        stalls = ", ".join([f"{s.reason} ({s.pct:.1f}%)" for s in self.performance_metrics.warp_stall_reasons[:3]]) or "None"
        spills = f"{self.launch_config.spill_loads_bytes}B loads, {self.launch_config.spill_stores_bytes}B stores"
        
        prompt = (
            f"[PROFILER REPORT]\n"
            f"Device: {self.hardware.device} ({self.hardware.arch}, {self.hardware.num_sms} SMs)\n"
            f"Kernel: {self.kernel_name} ({self.language.upper()})\n"
            f"Grid: {self.launch_config.grid} | Block: {self.launch_config.block}\n"
            f"Regs/Thread: {self.launch_config.registers_per_thread} | Shared Mem/Block: {self.launch_config.shared_mem_per_block_kb:.1f} KB | Local Spills: {spills}\n"
            f"Latency: {self.performance_metrics.latency_us:.2f} us\n"
            f"Speed of Light: Compute {self.performance_metrics.sol_compute_pct:.1f}% | Memory (DRAM) {self.performance_metrics.sol_memory_dram_pct:.1f}%\n"
            f"Occupancy: Achieved {self.performance_metrics.achieved_occupancy_pct:.1f}% / Theoretical {self.performance_metrics.theoretical_occupancy_pct:.1f}% (Limiter: {self.performance_metrics.occupancy_limiter})\n"
            f"Memory Coalescing: Load {self.performance_metrics.memory_efficiency.global_load_efficiency_pct:.1f}% | Store {self.performance_metrics.memory_efficiency.global_store_efficiency_pct:.1f}% | Bank Conflicts: {self.performance_metrics.memory_efficiency.shared_memory_bank_conflicts}\n"
            f"Top Warp Stalls: {stalls}\n"
            f"[/PROFILER REPORT]"
        )
        return prompt

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PIRReport:
        """Deserialize from dictionary."""
        hw = HardwareInfo(**data.get("hardware", {}))
        launch = LaunchConfig(**data.get("launch_config", {}))
        
        perf_data = data.get("performance_metrics", {})
        stalls = [WarpStallReason(**s) for s in perf_data.get("warp_stall_reasons", [])]
        mem_eff = MemoryEfficiency(**perf_data.get("memory_efficiency", {}))
        
        perf = PerformanceMetrics(
            latency_us=perf_data.get("latency_us", 0.0),
            sol_compute_pct=perf_data.get("sol_compute_pct", 0.0),
            sol_memory_dram_pct=perf_data.get("sol_memory_dram_pct", 0.0),
            achieved_occupancy_pct=perf_data.get("achieved_occupancy_pct", 0.0),
            theoretical_occupancy_pct=perf_data.get("theoretical_occupancy_pct", 0.0),
            occupancy_limiter=perf_data.get("occupancy_limiter", "None"),
            warp_stall_reasons=stalls,
            memory_efficiency=mem_eff
        )
        
        diag_data = data.get("diagnosis")
        diag = BottleneckDiagnosis(**diag_data) if diag_data else None
        
        return cls(
            kernel_name=data.get("kernel_name", "unknown"),
            language=data.get("language", "cuda"),
            hardware=hw,
            launch_config=launch,
            performance_metrics=perf,
            diagnosis=diag
        )

