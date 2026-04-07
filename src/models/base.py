"""
모델 인터페이스 정의
모든 모델 wrapper는 이 베이스 클래스를 상속하여 구현한다.
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """LM 모델 베이스 클래스"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """프롬프트를 입력받아 생성된 텍스트를 반환"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """모델 이름 반환"""
        pass
