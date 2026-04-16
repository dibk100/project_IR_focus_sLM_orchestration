from src.tasks.humaneval import HumanEvalSample
from src.tasks.mbpp import MBPPSample
from src.tasks.bigcode import BigCodeSample

from src.utils.prompting.planner_coder import *


def build_planner_prompt_for_sample(sample) -> str:
    """
    sample 타입에 따라 planner prompt를 선택한다.
    """
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_planner_prompt(sample)

    if isinstance(sample, MBPPSample):
        return build_mbpp_planner_prompt(sample)

    if isinstance(sample, BigCodeSample):
        return build_bigcode_planner_prompt(sample)

    raise TypeError(f"Unsupported sample type: {type(sample)}")


def build_coder_prompt_for_sample(sample, planner_output: str) -> str:
    """
    sample 타입에 따라 coder prompt를 선택한다.
    """
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_coder_prompt(sample, planner_output)

    if isinstance(sample, MBPPSample):
        return build_mbpp_coder_prompt(sample, planner_output)

    if isinstance(sample, BigCodeSample):
        return build_bigcode_coder_prompt(sample, planner_output)

    raise TypeError(f"Unsupported sample type: {type(sample)}")
