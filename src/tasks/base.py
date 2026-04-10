from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List

T = TypeVar("T")

class BaseTask(ABC, Generic[T]):
    @abstractmethod
    def load(self) -> List[T]:
        pass

    @abstractmethod
    def get_sample(self, index: int) -> T:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass