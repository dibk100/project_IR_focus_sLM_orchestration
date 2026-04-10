"""
Dataset adapter base interface

역할:
- sample -> prompt
- raw generation -> executable code
- code execution
- execution result -> orchestration-friendly dict
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AttemptRecord:
    """
    orchestration 계층이 공통으로 다룰 수 있는 attempt 결과 포맷
    """
    dataset: str
    task_id: str
    method: str
    model_name: str
    attempt_idx: int

    prompt: str
    raw_output: str
    generated_code: str

    status: str                 # "pass" / "fail" / "timeout"
    passed: bool
    exec_success: bool
    latency_sec: float

    error_type: Optional[str] = None
    error_message: Optional[str] = None

    meta: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """
    dataset-specific adapter interface

    orchestration은 이 인터페이스만 믿고 동작
    """

    dataset_name: str

    @abstractmethod
    def build_initial_prompt(self, sample: Any) -> str:
        """
        sample을 받아 첫 generation용 prompt를 만듦
        """
        raise NotImplementedError

    @abstractmethod
    def extract_code(self, sample: Any, raw_output: str) -> str:
        """
        모델 raw output에서 실행 가능한 code를 추출한다.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, sample: Any, code: str) -> Any:
        """
        dataset-specific execution을 수행한다.
        반환 타입은 dataset별 executor result 객체.
        """
        raise NotImplementedError

    @abstractmethod
    def classify_execution(self, exec_result: Any) -> Dict[str, Any]:
        """
        execution 결과를 공통 메타 정보로 변환한다.

        최소 반환 필드:
        {
            "status": str,
            "passed": bool,
            "exec_success": bool,
            "error_type": str | None,
            "error_message": str | None,
            "meta": dict,
        }
        """
        raise NotImplementedError

    def make_attempt_record(
        self,
        *,
        sample: Any,
        method: str,
        model_name: str,
        attempt_idx: int,
        prompt: str,
        raw_output: str,
        generated_code: str,
        latency_sec: float,
        exec_result: Any,
    ) -> AttemptRecord:
        """
        execution 결과를 공통 AttemptRecord로 변환한다.
        """
        info = self.classify_execution(exec_result)

        return AttemptRecord(
            dataset=self.dataset_name,
            task_id=sample.task_id,
            method=method,
            model_name=model_name,
            attempt_idx=attempt_idx,
            prompt=prompt,
            raw_output=raw_output,
            generated_code=generated_code,
            status=info["status"],
            passed=info["passed"],
            exec_success=info["exec_success"],
            latency_sec=latency_sec,
            error_type=info.get("error_type"),
            error_message=info.get("error_message"),
            meta=info.get("meta", {}),
        )