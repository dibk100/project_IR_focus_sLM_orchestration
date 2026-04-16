# src/tasks/bigcode.py

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .base import BaseTask


@dataclass
class BigCodeSample:
    task_id: str
    instruct_prompt: str
    entry_point: str
    test: str
    canonical_solution: str


class BigCodeTask(BaseTask[BigCodeSample]):
    def __init__(self, data_path: str = None):
        if data_path is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            data_path = os.path.join(
                project_root, "datasets", "bigcode", "raw", "bigcode.jsonl"
            )

        self.data_path = data_path
        self.samples: List[BigCodeSample] = []
        self._load_data()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line.strip())
                self.samples.append(
                    BigCodeSample(
                        task_id=raw["task_id"],
                        instruct_prompt=raw["instruct_prompt"],
                        entry_point=raw["entry_point"],
                        test=raw["test"],
                        canonical_solution=raw["canonical_solution"],
                    )
                )

    def load(self) -> List[BigCodeSample]:
        return self.samples

    def get_sample(self, index: int) -> BigCodeSample:
        return self.samples[index]

    def get_sample_by_id(self, task_id: str) -> BigCodeSample:
        for sample in self.samples:
            if sample.task_id == task_id:
                return sample
        raise ValueError(f"task_id '{task_id}' not found")

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"BigCodeTask(samples={len(self.samples)}))"