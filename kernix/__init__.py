"""
Kernix: A Profiler-Grounded GPU Kernel Optimization Engine
"""

from kernix.profiler.correctness_gate import CorrectnessGate
from kernix.profiler.real_executor import RealKernelExecutor
from kernix.profiler.ncu_parser import NCUParser
from kernix.schema.pir import PIRReport, HardwareInfo, LaunchConfig, PerformanceMetrics

__version__ = "1.0.0"

__all__ = [
    "CorrectnessGate",
    "RealKernelExecutor",
    "NCUParser",
    "PIRReport",
    "HardwareInfo",
    "LaunchConfig",
    "PerformanceMetrics",
]
