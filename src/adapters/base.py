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
    ver3 기준 필드를 포함하되, 기존 exec_success도 함께 유지한다.
    """
    dataset: str
    task_id: str
    method: str
    model_name: str
    attempt_idx: int

    prompt: str
    raw_output: str
    generated_code: str

    status: str                 # "PASS" / "EXEC_FAIL:TypeError" / "TEST_FAIL:AssertionError"
    passed: bool
    exec_ok: bool
    test_pass: bool
    latency_sec: float

    # ver2 호환용
    exec_success: bool

    error_type: Optional[str] = None
    error_stage: Optional[str] = None
    error_message: Optional[str] = None

    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None

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
            "exec_ok": bool,
            "test_pass": bool,
            "exec_success": bool,   # ver2 호환용 (보통 exec_ok와 동일)
            "error_type": str | None,
            "error_stage": str | None,   # "exec" / "test" / None
            "error_message": str | None,
            "tests_passed": int | None,
            "tests_total": int | None,
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

        exec_ok = info.get("exec_ok", info.get("exec_success", False))
        test_pass = info.get("test_pass", info.get("passed", False))

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
            exec_ok=exec_ok,
            test_pass=test_pass,
            latency_sec=latency_sec,
            exec_success=info.get("exec_success", exec_ok),  # ver2 호환용
            error_type=info.get("error_type"),
            error_stage=info.get("error_stage"),
            error_message=info.get("error_message"),
            tests_passed=info.get("tests_passed"),
            tests_total=info.get("tests_total"),
            meta=info.get("meta", {}),
        )

    @abstractmethod
    def build_repair_prompt(
        self,
        sample: Any,
        previous_code: str,
        error_message: str | None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_refinement_prompt(
        self,
        sample: Any,
        previous_code: str,
    ) -> str:
        raise NotImplementedError

    def extract_code_for_planner(
        self,
        sample: Any,
        raw_output: str,
    ) -> str:
        raise NotImplementedError