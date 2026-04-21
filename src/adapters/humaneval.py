# src/adapters/humaneval.py
"""
HumanEval adapter
"""
from typing import Any, Dict

from src.adapters.base import BaseAdapter
from src.tasks.humaneval import HumanEvalSample
from src.evaluation.executor import execute_humaneval, HumanEvalExecutionTrace
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

    def execute(self, sample: HumanEvalSample, code: str) -> HumanEvalExecutionTrace:
        return execute_humaneval(
            code=code,
            test=sample.test,
            entry_point=sample.entry_point,
        )

    def extract_code_for_planner(self, sample, raw_output: str) -> str:
        return extract_humaneval_full_function_code(
            raw_output=raw_output,
            entry_point=sample.entry_point,
            fallback_prompt=sample.prompt,
        )