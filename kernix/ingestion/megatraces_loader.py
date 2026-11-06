"""Loader and Normalizer for KernelBench Mega Traces (Infatoshi/kernelbench-mega-traces).

Parses autonomous agent sessions for low-level GPU kernel generation and megakernel fusion.
Extracts multi-round refinement trajectories, compiler diagnostic feedback, and speedup steps.
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Optional, Tuple


class MegaTracesLoader:
    """Parses agent conversation logs into before/after optimization steps."""

    @staticmethod
    def extract_code_from_content(content: str) -> List[Tuple[str, str]]:
        """Extract all code blocks with their detected language from markdown text."""
        pattern = r"```(triton|python|cuda|cpp)?\s*(.*?)\s*```"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        results = []
        for lang, code in matches:
            clean_lang = lang.lower() if lang else "triton"
            if clean_lang in ["python", "triton", "cuda", "cpp"] and len(code.strip()) > 50:
                results.append((clean_lang, code.strip()))
        return results

    @classmethod
    def parse_session_file(cls, session_path: str) -> List[Dict[str, Any]]:
        """Extract optimization turns from an individual agent session .jsonl file."""
        records = []
        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return cls.parse_session_records(records, session_id=session_path)

    @classmethod
    def parse_session_records(cls, records: List[Dict[str, Any]], session_id: str = "unknown") -> List[Dict[str, Any]]:
        """Extract step-by-step kernel refinement pairs from parsed session records."""
        trajectories = []
        current_code: Optional[str] = None
        current_lang: str = "triton"
        last_feedback: str = ""

        for record in records:
            role = record.get("role", "")
            content = record.get("content", "")

            # If agent outputs code
            if role in ["assistant", "agent", "model"]:
                code_blocks = cls.extract_code_from_content(content)
                if code_blocks:
                    lang, new_code = code_blocks[-1]
                    if current_code and new_code != current_code:
                        # We have a transition (current_code -> new_code)
                        trajectories.append({
                            "source": "Infatoshi/kernelbench-mega-traces",
                            "session_id": session_id,
                            "step_index": len(trajectories) + 1,
                            "language": lang,
                            "initial_code": current_code,
                            "refactored_code": new_code,
                            "feedback_or_error": last_feedback[:800],
                            "has_compiler_error": "error" in last_feedback.lower() or "traceback" in last_feedback.lower()
                        })
                    current_code = new_code
                    current_lang = lang

            # If environment provides feedback
            elif role in ["tool", "system", "user", "environment"]:
                last_feedback = content

        return trajectories
