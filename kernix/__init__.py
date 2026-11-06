"""
Kernix: A Profiler-Grounded GPU Kernel Optimization Engine
"""
from gpuperf.profiler.correctness_gate import CorrectnessGate
from gpuperf.profiler.real_executor import RealKernelExecutor
from gpuperf.profiler.ncu_parser import NCUParser
from gpuperf.schema.pir import PIRReport, HardwareInfo, LaunchConfig, PerformanceMetrics

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
