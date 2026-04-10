"""
MBPP adapter
"""
from typing import Any, Dict

from src.adapters.base import BaseAdapter
from src.tasks.mbpp import MBPPSample
from src.utils.prompting import build_mbpp_prompt, extract_mbpp_code
from src.evaluation.executor import execute_mbpp_staged, MBPPExecutionTrace


class MBPPAdapter(BaseAdapter):
    dataset_name = "mbpp"

    def build_initial_prompt(self, sample: MBPPSample) -> str:
        return build_mbpp_prompt(sample)

    def extract_code(self, sample: MBPPSample, raw_output: str) -> str:
        return extract_mbpp_code(raw_output)

    def execute(self, sample: MBPPSample, code: str) -> MBPPExecutionTrace:
        return execute_mbpp_staged(
            code=code,
            test_list=sample.test_list,
            test_setup_code=sample.test_setup_code,
        )

    def classify_execution(self, exec_result: MBPPExecutionTrace) -> Dict[str, Any]:
        """
        MBPP staged execution 결과를 공통 포맷으로 변환

        execution success 정의:
        - code_exec_passed and setup_exec_passed

        semantic failure:
        - failed_stage == "test" and error_type == "AssertionError"
        """
        if exec_result.passed:
            return {
                "status": "pass",
                "passed": True,
                "exec_success": True,
                "error_type": None,
                "error_message": None,
                "meta": {
                    "code_exec_passed": True,
                    "setup_exec_passed": True,
                    "test_exec_passed": True,
                    "failed_stage": None,
                    "failed_test_index": None,
                },
            }

        error_type = self._normalize_error_type(exec_result.error_type, exec_result.failed_stage)
        exec_success = exec_result.code_exec_passed and exec_result.setup_exec_passed

        return {
            "status": "fail",
            "passed": False,
            "exec_success": exec_success,
            "error_type": error_type,
            "error_message": exec_result.error,
            "meta": {
                "code_exec_passed": exec_result.code_exec_passed,
                "setup_exec_passed": exec_result.setup_exec_passed,
                "test_exec_passed": exec_result.test_exec_passed,
                "failed_stage": exec_result.failed_stage,
                "failed_test_index": exec_result.failed_test_index,
                "raw_error_type": exec_result.error_type,
            },
        }

    def _normalize_error_type(self, error_type: str | None, failed_stage: str | None) -> str:
        """
        MBPP는 staged execution이므로 failed_stage 정보까지 반영해서 해석한다.
        """
        if failed_stage == "code":
            if error_type == "EmptyGeneration":
                return "empty_generation"
            if error_type == "SyntaxError":
                return "syntax_error"
            if error_type == "IndentationError":
                return "indentation_error"
            return "code_error"

        if failed_stage == "setup":
            if error_type == "ImportError" or error_type == "ModuleNotFoundError":
                return "import_error"
            return "setup_error"

        if failed_stage == "test":
            if error_type == "AssertionError":
                return "assertion_error"
            if error_type == "NameError":
                return "name_error"
            if error_type == "TypeError":
                return "type_error"
            if error_type == "IndexError":
                return "index_error"
            if error_type == "KeyError":
                return "key_error"
            if error_type == "ValueError":
                return "value_error"
            if error_type == "AttributeError":
                return "attribute_error"
            return "test_execution_error"

        return "unknown_error"