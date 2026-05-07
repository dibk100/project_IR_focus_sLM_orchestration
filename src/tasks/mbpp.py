# src/tasks/mbpp.py
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .base import BaseTask


@dataclass
class MBPPSample:
    task_id: str
    problem_text: str
    test_list: List[str]
    test_setup_code: str = ""
    challenge_test_list: List[str] = field(default_factory=list)
    reference_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MBPPTask(BaseTask[MBPPSample]):
    def __init__(self, data_path: str = None):
        if data_path is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            data_path = os.path.join(
                project_root, "datasets", "mbpp_sanitized", "raw", "mbpp_sanitized.jsonl"
            )

        self.data_path = Path(data_path)
        self.samples: List[MBPPSample] = []
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)

                problem_text = raw.get("prompt") or raw.get("text")
                if problem_text is None:
                    raise ValueError(f"MBPP sample {raw.get('task_id')} has no prompt/text field.")

                test_setup_code = raw.get("test_setup_code", "") or ""

                # MBPP sanitized uses test_imports instead of test_setup_code.
                if "test_imports" in raw and raw["test_imports"]:
                    test_setup_code = "\n".join(raw["test_imports"])

                self.samples.append(
                    MBPPSample(
                        task_id=str(raw["task_id"]),
                        problem_text=problem_text.strip(),
                        test_list=raw.get("test_list", []),
                        test_setup_code=test_setup_code,
                        challenge_test_list=raw.get("challenge_test_list", []) or [],
                        reference_code=raw.get("code"),
                        metadata={
                            "dataset": "mbpp_sanitized"
                            if "prompt" in raw else "mbpp"
                        },
                    )
                )

    def load(self) -> List[MBPPSample]:
        return self.samples

    def get_sample(self, index: int) -> MBPPSample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"MBPPTask(samples={len(self.samples)})"