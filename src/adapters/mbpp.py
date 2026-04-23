# src/adapters/mbpp.py
"""
MBPP adapter
"""
from typing import Any, Dict

from src.adapters.base import BaseAdapter
from src.tasks.mbpp import MBPPSample
from src.evaluation.executor import execute_mbpp_staged, MBPPExecutionTrace
from src.utils.prompting import (
    build_mbpp_prompt,
    build_mbpp_repair_prompt,
    extract_mbpp_code,
    build_mbpp_refinement_prompt,
)


class MBPPAdapter(BaseAdapter):
    dataset_name = "mbpp"

    def build_initial_prompt(self, sample: MBPPSample) -> str:
        return build_mbpp_prompt(sample)
    
    def build_repair_prompt(self, sample, previous_code, error_message):
        return build_mbpp_repair_prompt(
            sample=sample,
            previous_code=previous_code,
            error_message=error_message,
        )
        
    def build_refinement_prompt(
        self,
        sample,
        previous_code: str,
    ) -> str:
        return build_mbpp_refinement_prompt(
            sample=sample,
            previous_code=previous_code,
        )

    def extract_code(self, sample: MBPPSample, raw_output: str) -> str:
        return extract_mbpp_code(raw_output)

    def execute(self, sample: MBPPSample, code: str) -> MBPPExecutionTrace:
        return execute_mbpp_staged(
            code=code,
            test_list=sample.test_list,
            test_setup_code=sample.test_setup_code,
        )

    def extract_code_for_planner(self, sample, raw_output: str) -> str:
        return self.extract_code(sample, raw_output)