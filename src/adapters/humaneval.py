"""
HumanEval adapter
"""
from typing import Any, Dict

from src.adapters.base import BaseAdapter
from src.tasks.humaneval import HumanEvalSample
from src.evaluation.executor import execute_humaneval, ExecutionResult
from src.utils.prompting import (
    build_humaneval_prompt,
    build_humaneval_repair_prompt,
    extract_humaneval_code,
    build_humaneval_refinement_prompt,
    extract_humaneval_full_function_code
)

class HumanEvalAdapter(BaseAdapter):
    dataset_name = "humaneval"

    def build_initial_prompt(self, sample: HumanEvalSample) -> str:
        return build_humaneval_prompt(sample)
    
    def build_repair_prompt(self, sample, previous_code, error_message):
        return build_humaneval_repair_prompt(
            sample=sample,
            previous_code=previous_code,
            error_message=error_message,
        )
    
    def build_refinement_prompt(
        self,
        sample,
        previous_code: str,
    ) -> str:
        return build_humaneval_refinement_prompt(
            sample=sample,
            previous_code=previous_code,
        )
    
    def extract_code(self, sample: HumanEvalSample, raw_output: str) -> str:
        return extract_humaneval_code(sample, raw_output)

    def execute(self, sample: HumanEvalSample, code: str) -> ExecutionResult:
        return execute_humaneval(
            code=code,
            test=sample.test,
            entry_point=sample.entry_point,
        )

    def classify_execution(self, exec_result: ExecutionResult) -> Dict[str, Any]:
        """
        HumanEval execution 결과를 ver3 공통 포맷으로 변환한다.

        상태 규칙:
        - passed=True                     -> PASS
        - timeout=True                    -> EXEC_FAIL:Timeout
        - AssertionError                  -> TEST_FAIL:AssertionError
        - 그 외 실행 예외                 -> EXEC_FAIL:<ErrorType>
        """
        if exec_result.passed:
            return {
                "status": "PASS",
                "passed": True,
                "exec_ok": True,
                "test_pass": True,
                "error_type": None,
                "error_stage": None,
                "error_message": None,
                "tests_passed": None,
                "tests_total": None,
                "meta": {
                    "timeout": False,
                    "raw_error_type": None,
                },
            }

        if exec_result.timeout:
            return {
                "status": "EXEC_FAIL:Timeout",
                "passed": False,
                "exec_ok": False,
                "test_pass": False,
                "error_type": "Timeout",
                "error_stage": "exec",
                "error_message": exec_result.error,
                "tests_passed": None,
                "tests_total": None,
                "meta": {
                    "timeout": True,
                    "raw_error_type": exec_result.error_type,
                },
            }

        error_type = self._normalize_error_type(exec_result.error_type, exec_result.error)

        if error_type == "AssertionError":
            return {
                "status": "TEST_FAIL:AssertionError",
                "passed": False,
                "exec_ok": True,
                "test_pass": False,
                "error_type": "AssertionError",
                "error_stage": "test",
                "error_message": exec_result.error,
                "tests_passed": None,
                "tests_total": None,
                "meta": {
                    "timeout": False,
                    "raw_error_type": exec_result.error_type,
                },
            }

        return {
            "status": f"EXEC_FAIL:{error_type}",
            "passed": False,
            "exec_ok": False,
            "test_pass": False,
            "error_type": error_type,
            "error_stage": "exec",
            "error_message": exec_result.error,
            "tests_passed": None,
            "tests_total": None,
            "meta": {
                "timeout": False,
                "raw_error_type": exec_result.error_type,
            },
        }

    def _normalize_error_type(self, error_type: str | None, error_message: str | None) -> str:
        if error_type:
            lowered = error_type.lower()
            if lowered == "syntaxerror":
                return "SyntaxError"
            if lowered == "assertionerror":
                return "AssertionError"
            if lowered == "typeerror":
                return "TypeError"
            if lowered == "nameerror":
                return "NameError"
            if lowered == "indexerror":
                return "IndexError"
            if lowered == "keyerror":
                return "KeyError"
            if lowered == "valueerror":
                return "ValueError"
            if lowered == "attributeerror":
                return "AttributeError"
            if lowered in ("importerror", "modulenotfounderror"):
                return "ImportError"

        if error_message:
            lowered = error_message.lower()
            if "syntaxerror" in lowered:
                return "SyntaxError"
            if "assertionerror" in lowered:
                return "AssertionError"
            if "typeerror" in lowered:
                return "TypeError"
            if "nameerror" in lowered:
                return "NameError"
            if "indexerror" in lowered:
                return "IndexError"
            if "keyerror" in lowered:
                return "KeyError"
            if "valueerror" in lowered:
                return "ValueError"
            if "attributeerror" in lowered:
                return "AttributeError"
            if "importerror" in lowered or "modulenotfounderror" in lowered:
                return "ImportError"

        return "RuntimeError"
    
    def extract_code_for_planner(self, sample, raw_output: str) -> str:
        return extract_humaneval_full_function_code(
            raw_output=raw_output,
            entry_point=sample.entry_point,
            fallback_prompt=sample.prompt,
        )