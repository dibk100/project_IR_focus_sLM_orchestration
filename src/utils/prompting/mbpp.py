# src/utils/prompting/mbpp.py

"""
MBPP 전용 프롬프트 및 코드 추출 유틸리티
"""

from src.tasks.mbpp import MBPPSample
from .common import strip_code_fence

def build_mbpp_prompt(sample: MBPPSample) -> str:
    """
    MBPP는 자연어 문제 설명을 바탕으로
    실행 가능한 전체 Python 코드를 생성하도록 유도한다.

    중요:
    - 설명 금지
    - markdown fence 금지
    - placeholder 금지
    - 필요한 helper class / helper function 포함
    """
    return (
        "You are a Python programmer.\n"
        "Write a complete Python solution for the following problem.\n"
        "Output only executable Python code.\n"
        "Do not include any explanation, markdown fences, examples, or comments like 'Your code here'.\n"
        "Your code must include all necessary definitions such as helper functions or classes.\n"
        "Do not leave the function body empty.\n\n"
        f"Problem:\n{sample.problem_text}\n"
    )


def extract_mbpp_code(raw_output: str) -> str:
    """
    MBPP는 모델이 전체 코드를 생성해야 하므로
    generation 자체를 정리해서 그대로 사용한다.
    """
    return strip_code_fence(raw_output)

# 좀 더 단순한 프롬프트
# def build_mbpp_prompt(sample: MBPPSample) -> str:
#     """
#     MBPP는 자연어 문제 설명을 바탕으로
#     실행 가능한 전체 Python 코드를 생성하도록 유도한다.

#     중요:
#     - 함수만이 아니라 필요한 helper class / helper function까지
#       모두 포함한 '전체 코드'를 생성해야 할 수 있다.
#     - 따라서 prompt에서 'all necessary definitions'를 명시한다.
#     """
#     return (
#         "Write Python code to solve the following problem.\n"
#         "Return only code.\n"
#         "Your code must include all necessary definitions, "
#         "such as helper functions or classes.\n\n"
#         f"Problem:\n{sample.problem_text}\n"
#     )
