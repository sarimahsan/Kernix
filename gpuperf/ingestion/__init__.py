"""Data loaders for public GPU kernel benchmarks."""
from gpuperf.ingestion.kernelbenchx_loader import KernelBenchXLoader
from gpuperf.ingestion.megatraces_loader import MegaTracesLoader

__all__ = ["KernelBenchXLoader", "MegaTracesLoader"]

