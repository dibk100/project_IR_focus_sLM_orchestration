"""
BigCode adapter
"""
from typing import Any, Dict

from src.adapters.base import BaseAdapter
from src.tasks.bigcode import BigCodeSample
from src.evaluation.executor import execute_bigcode, BigCodeExecutionTrace
from src.utils.prompting import (
    build_bigcode_prompt,
    build_bigcode_repair_prompt,
    extract_bigcode_code,
    build_bigcode_refinement_prompt,
    extract_bigcode_full_function_code
)

class BigCodeAdapter(BaseAdapter):
    dataset_name = "bigcode"

    def build_initial_prompt(self, sample: BigCodeSample) -> str:
        return build_bigcode_prompt(sample)
    
    def build_repair_prompt(self, sample, previous_code, error_message):
        return build_bigcode_repair_prompt(
            sample=sample,
            previous_code=previous_code,
            error_message=error_message,
        )
    
    def build_refinement_prompt(
        self,
        sample,
        previous_code: str,
    ) -> str:
        return build_bigcode_refinement_prompt(
            sample=sample,
            previous_code=previous_code,
        )
    
    def extract_code(self, sample: BigCodeSample, raw_output: str) -> str:
        return extract_bigcode_code(raw_output)

    def execute(self, sample: BigCodeSample, code: str) -> BigCodeExecutionTrace:
        return execute_bigcode(
            code=code,
            test=sample.test,
        )
    
    def extract_code_for_planner(self, sample, raw_output: str) -> str:
        return extract_bigcode_full_function_code(
            raw_output=raw_output,
            entry_point=sample.entry_point,
            fallback_prompt=sample.instruct_prompt,
        )