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
    input: str
    test: str
    hint: str = ""
    test_list: List[str] = field(default_factory=list)
    test_setup_code: str = ""
    challenge_test_list: List[str] = field(default_factory=list)
    reference_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def problem_text(self) -> str:
        return self.input


class MBPPTask(BaseTask[MBPPSample]):
    def __init__(self, data_path: str = None):
        if data_path is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            data_path = os.path.join(
                project_root,
                "datasets",
                "mbpp_sanitized",
                "raw",
                "mbpp_sanitized.jsonl",
            )

        self.data_path = Path(data_path)
        self.samples: List[MBPPSample] = []
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line.strip())

                problem_text = raw.get("prompt") or raw.get("text")
                if problem_text is None:
                    raise ValueError(
                        f"MBPP sample {raw.get('task_id')} has no prompt/text field."
                    )

                test_setup_code = raw.get("test_setup_code", "") or ""

                if raw.get("test_imports"):
                    test_setup_code = "\n".join(raw["test_imports"])

                test_list = raw.get("test_list", []) or []
                test_hint = test_list[0] if test_list else ""

                test_code = "\n".join(
                    [test_setup_code] + test_list
                    if test_setup_code
                    else test_list
                )

                self.samples.append(
                    MBPPSample(
                        task_id=str(raw["task_id"]),
                        input=problem_text.strip(),
                        test=test_code,
                        hint=test_hint,
                        test_list=test_list,
                        test_setup_code=test_setup_code,
                        challenge_test_list=raw.get("challenge_test_list", []) or [],
                        reference_code=raw.get("code"),
                        metadata={
                            "dataset": "mbpp_sanitized"
                            if "prompt" in raw
                            else "mbpp"
                        },
                    )
                )

    def load(self) -> List[MBPPSample]:
        return self.samples

    def get_sample(self, index: int) -> MBPPSample:
        return self.samples[index]

    def get_sample_by_id(self, task_id: str) -> MBPPSample:
        for sample in self.samples:
            if sample.task_id == task_id:
                return sample
        raise ValueError(f"task_id '{task_id}' not found")

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"MBPPTask(samples={len(self.samples)})"