# src/adapters/base.py
"""
Dataset adapter base interface

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

    status: str                # "PASS" / "EXEC_FAIL:<Type>" / "DEFINE_TEST_FAIL:<Type>" / "TEST_FAIL:<Type>" / "OTHER_FAIL:<Type>"
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

    def classify_execution(self, exec_result: Any) -> Dict[str, Any]:
        """
        dataset별 execution 결과를 공통 포맷으로 변환한다.

        stage 정의:
        - "code"         : generated code 자체가 exec 실패
        - "define_test"  : test/check/unittest 정의 코드 exec 실패
        - "run_test"     : 테스트 실행 실패

        status 규칙:
        - passed=True                          -> PASS
        - failed_stage == "code"              -> EXEC_FAIL:<ErrorType>
        - failed_stage == "define_test"       -> DEFINE_TEST_FAIL:<ErrorType>
        - failed_stage == "run_test"          -> TEST_FAIL:<ErrorType>
        - 그 외                               -> OTHER_FAIL:<ErrorType>
        """

        # -------------------------
        # 1. raw 값 추출
        # -------------------------
        failed_stage = getattr(exec_result, "failed_stage", None)
        raw_error_type = getattr(exec_result, "error_type", None)
        error_message = getattr(exec_result, "error", None)

        code_exec_passed = getattr(exec_result, "code_exec_passed", False)
        setup_exec_passed = getattr(exec_result, "setup_exec_passed", False)
        test_exec_passed = getattr(exec_result, "test_exec_passed", False)
        failed_test_index = getattr(exec_result, "failed_test_index", None)

        # -------------------------
        # 2. meta (raw 정보 보존)
        # -------------------------
        base_meta = {
            "code_exec_passed": code_exec_passed,
            "setup_exec_passed": setup_exec_passed,
            "test_exec_passed": test_exec_passed,
            "failed_stage": failed_stage,
            "failed_test_index": failed_test_index,
            "raw_error_type": raw_error_type,
        }

        # -------------------------
        # 3. PASS
        # -------------------------
        if exec_result.passed:
            return {
                "status": "PASS",
                "passed": True,
                "exec_ok": True,
                "test_pass": True,
                "exec_success": True,
                "error_type": None,
                "error_stage": None,
                "error_message": None,
                "tests_passed": None,
                "tests_total": None,
                "meta": base_meta,
            }

        # -------------------------
        # 4. 기본 값
        # -------------------------
        error_type = raw_error_type or "UnknownError"
        exec_ok = code_exec_passed and setup_exec_passed

        # -------------------------
        # 5. code 실패
        # -------------------------
        if failed_stage == "code":
            return {
                "status": f"EXEC_FAIL:{error_type}",
                "passed": False,
                "exec_ok": False,
                "test_pass": False,
                "exec_success": False,
                "error_type": error_type,
                "error_stage": "exec",
                "error_message": error_message,
                "tests_passed": None,
                "tests_total": None,
                "meta": base_meta,
            }

        # -------------------------
        # 6. define_test 실패
        # -------------------------
        if failed_stage == "define_test":
            return {
                "status": f"DEFINE_TEST_FAIL:{error_type}",
                "passed": False,
                "exec_ok": False,
                "test_pass": False,
                "exec_success": False,
                "error_type": error_type,
                "error_stage": "exec",
                "error_message": error_message,
                "tests_passed": None,
                "tests_total": None,
                "meta": base_meta,
            }

        # -------------------------
        # 7. run_test 실패
        # -------------------------
        if failed_stage == "run_test":
            return {
                "status": f"TEST_FAIL:{error_type}",
                "passed": False,
                "exec_ok": exec_ok,
                "test_pass": False,
                "exec_success": exec_ok,
                "error_type": error_type,
                "error_stage": "test",
                "error_message": error_message,
                "tests_passed": failed_test_index,
                "tests_total": None,
                "meta": base_meta,
            }

        # -------------------------
        # 8. fallback (unexpected)
        # -------------------------
        return {
            "status": f"OTHER_FAIL:{error_type}",
            "passed": False,
            "exec_ok": False,
            "test_pass": False,
            "exec_success": False,
            "error_type": error_type,
            "error_stage": "unknown",
            "error_message": error_message,
            "tests_passed": None,
            "tests_total": None,
            "meta": {
                **base_meta,
                "unexpected": True,
                "unexpected_reason": f"Unhandled failed_stage: {failed_stage}",
            },
        }

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