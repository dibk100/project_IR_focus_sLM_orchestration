"""
Task 인터페이스 정의
모든 task loader는 이 베이스 클래스를 상속하여 구현한다.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TaskSample:
    """단일 태스크 샘플"""
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: Optional[str] = None
    test: Optional[str] = None


class BaseTask(ABC):
    """태스크 로더 베이스 클래스"""

    @abstractmethod
    def load(self) -> List[TaskSample]:
        """전체 데이터셋 로드"""
        pass

    @abstractmethod
    def get_sample(self, index: int) -> TaskSample:
        """인덱스로 단일 샘플 반환"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
