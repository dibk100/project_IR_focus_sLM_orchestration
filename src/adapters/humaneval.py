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
        HumanEval execution 결과를 공통 포맷으로 변환

        현재 HumanEval executor는 staged execution이 아니므로
        practical rule을 사용한다:
        - passed=True -> pass
        - timeout=True -> timeout
        - AssertionError -> exec_success=True, fail
        - 그 외 -> exec_success=False, fail
        """
        if exec_result.passed:
            return {
                "status": "pass",
                "passed": True,
                "exec_success": True,
                "error_type": None,
                "error_message": None,
                "meta": {
                    "timeout": False,
                },
            }

        if exec_result.timeout:
            return {
                "status": "timeout",
                "passed": False,
                "exec_success": False,
                "error_type": "timeout",
                "error_message": exec_result.error,
                "meta": {
                    "timeout": True,
                },
            }

        error_type = self._normalize_error_type(exec_result.error_type, exec_result.error)
        exec_success = exec_result.error_type == "AssertionError"

        return {
            "status": "fail",
            "passed": False,
            "exec_success": exec_success,
            "error_type": error_type,
            "error_message": exec_result.error,
            "meta": {
                "timeout": False,
                "raw_error_type": exec_result.error_type,
            },
        }

    def _normalize_error_type(self, error_type: str | None, error_message: str | None) -> str:
        if error_type:
            lowered = error_type.lower()
            if lowered == "syntaxerror":
                return "syntax_error"
            if lowered == "assertionerror":
                return "assertion_error"
            if lowered == "typeerror":
                return "type_error"
            if lowered == "nameerror":
                return "name_error"
            if lowered == "indexerror":
                return "index_error"
            if lowered == "keyerror":
                return "key_error"
            if lowered == "valueerror":
                return "value_error"
            if lowered == "attributeerror":
                return "attribute_error"
            if lowered in ("importerror", "modulenotfounderror"):
                return "import_error"

        if error_message:
            lowered = error_message.lower()
            if "syntaxerror" in lowered:
                return "syntax_error"
            if "assertionerror" in lowered:
                return "assertion_error"
            if "typeerror" in lowered:
                return "type_error"
            if "nameerror" in lowered:
                return "name_error"
            if "indexerror" in lowered:
                return "index_error"
            if "keyerror" in lowered:
                return "key_error"
            if "valueerror" in lowered:
                return "value_error"
            if "attributeerror" in lowered:
                return "attribute_error"
            if "importerror" in lowered or "modulenotfounderror" in lowered:
                return "import_error"

        return "runtime_error"
    
    def extract_code_for_planner(self, sample, raw_output: str) -> str:
        return extract_humaneval_full_function_code(
            raw_output=raw_output,
            entry_point=sample.entry_point,
            fallback_prompt=sample.prompt,
        )