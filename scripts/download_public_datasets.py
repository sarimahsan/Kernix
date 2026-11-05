"""Download and inspect public GPU kernel datasets from Hugging Face."""

import os
import json
from datasets import load_dataset


def download_datasets(output_dir: str = "./data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    print("=== Downloading Public Datasets ===")

    # 1. KernelBenchX
    print("\n1. Fetching KernelBenchX (BonnieWang/KernelBenchX)...")
    try:
        kbx = load_dataset("BonnieWang/KernelBenchX", "corpus_all")
        kbx_path = os.path.join(output_dir, "kernelbenchx.jsonl")
        kbx["train"].to_json(kbx_path)
        print(f"-> Successfully saved KernelBenchX to {kbx_path} ({len(kbx['train'])} records)")
    except Exception as e:
        print(f"-> Notice: Failed to fetch KernelBenchX: {e}")

    # 2. KernelBench Mega Traces
    print("\n2. Fetching KernelBench Mega Traces (Infatoshi/kernelbench-mega-traces)...")
    try:
        mega = load_dataset("Infatoshi/kernelbench-mega-traces")
        mega_path = os.path.join(output_dir, "mega_traces.jsonl")
        mega["train"].to_json(mega_path)
        print(f"-> Successfully saved Mega Traces to {mega_path} ({len(mega['train'])} records)")
    except Exception as e:
        print(f"-> Notice: Failed to fetch Mega Traces: {e}")

    # 3. KernelSight
    print("\n3. Fetching KernelSight (williamhtan/kernelsight)...")
    try:
        ks = load_dataset("williamhtan/kernelsight")
        print(f"-> Successfully loaded KernelSight: {ks}")
    except Exception as e:
        print(f"-> Notice: Failed to fetch KernelSight: {e}")

    print("\n=== Dataset Download Complete ===")


if __name__ == "__main__":
    download_datasets()

