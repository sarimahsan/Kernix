"""Dataset loaders and prompt formatters for GPUPerf fine-tuning."""

from __future__ import annotations
import json
from typing import List, Dict, Any


class TrajectoryDataset:
    """Utility for loading and validating SFT and DPO training files."""

    @staticmethod
    def load_sft_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Load messages list from an SFT jsonl file."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" in obj:
                        records.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_idx}: {e}")
        return records

    @staticmethod
    def load_dpo_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Load prompt/chosen/rejected triplets from a DPO jsonl file."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "prompt" in obj and "chosen" in obj and "rejected" in obj:
                        records.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_idx}: {e}")
        return records

