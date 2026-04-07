"""
HumanEval Task Loader
datasets/humaneval/raw/humaneval.jsonl에서 데이터를 로드한다.
"""
import os
import json
from typing import List

from src.tasks.base import BaseTask, TaskSample


class HumanEvalTask(BaseTask):
    """HumanEval 데이터셋 로더"""

    def __init__(self, data_path: str = None):
        if data_path is None:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            data_path = os.path.join(
                project_root, "datasets", "humaneval", "raw", "humaneval.jsonl"
            )
        self.data_path = data_path
        self.samples: List[TaskSample] = []
        self.load()

    def load(self) -> List[TaskSample]:
        """jsonl 파일에서 전체 데이터셋 로드"""
        self.samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                sample = TaskSample(
                    task_id=item["task_id"],
                    prompt=item["prompt"],
                    entry_point=item["entry_point"],
                    canonical_solution=item.get("canonical_solution"),
                    test=item.get("test"),
                )
                self.samples.append(sample)
        return self.samples

    def get_sample(self, index: int) -> TaskSample:
        """인덱스로 단일 샘플 반환"""
        return self.samples[index]

    def get_sample_by_id(self, task_id: str) -> TaskSample:
        """task_id로 샘플 검색 (예: 'HumanEval/0')"""
        for sample in self.samples:
            if sample.task_id == task_id:
                return sample
        raise ValueError(f"task_id '{task_id}' not found")

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"HumanEvalTask(samples={len(self.samples)})"
