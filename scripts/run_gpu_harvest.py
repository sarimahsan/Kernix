"""CLI Entry Point: 1-Hour Universal Multi-GPU Profiling Harvester.

Run this script on any machine with an NVIDIA GPU (T4, L4, A100, RTX 4090)
to harvest empirical performance profiles and optimization trajectories.

Usage:
    python scripts/run_gpu_harvest.py --duration-minutes 60
"""

import sys
import os
import argparse

# Ensure project root is in sys.path when invoked from scripts/ or anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpuperf.profiler.multi_gpu_harvester import MultiGPUHarvester


def main():
    parser = argparse.ArgumentParser(description="GPUPerf Multi-GPU Profiling Harvester")
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=60,
        help="Harvesting duration budget in minutes (default: 60)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Directory to store hardware-tagged raw harvest files"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Run in synthetic mock mode for local testing without an active NVIDIA GPU"
    )

    args = parser.parse_args()

    # Automatically check if GPU is available; fallback to mock if none
    use_mock = args.mock
    try:
        import torch
        if not torch.cuda.is_available():
            print("Note: No CUDA GPU detected on this host. Falling back to mock simulation mode.")
            use_mock = True
    except ImportError:
        use_mock = True

    MultiGPUHarvester.harvest_gpu_session(
        output_dir=args.output_dir,
        duration_minutes=args.duration_minutes,
        use_mock=use_mock
    )


if __name__ == "__main__":
    main()

