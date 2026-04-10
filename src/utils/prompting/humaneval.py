
# src/utils/prompting/humaneval.py

"""
HumanEval 전용 프롬프트 및 코드 추출 유틸리티
"""
from src.tasks.humaneval import HumanEvalSample
from .common import strip_code_fence, truncate_at_new_toplevel_block


def build_humaneval_prompt(sample: HumanEvalSample) -> str:
    """
    HumanEval은 원본 prompt 자체가
    함수 시그니처 + docstring completion 형식이라 그대로 사용
    """
    return sample.prompt


def extract_humaneval_code(sample: HumanEvalSample, generation: str) -> str:
    """
    HumanEval 생성 결과에서 함수 본문만 추출하고,
    원본 prompt와 합쳐 실행 가능한 전체 함수 코드 생성해야함.

    Args:
        prompt: 원본 프롬프트 (함수 시그니처 + docstring)
        generation: 모델 생성 텍스트
    
    Returns:
        prompt + 추출된 함수 본문 (실행 가능한 완전한 함수)
    """
    cleaned = strip_code_fence(generation)
    extracted_body = truncate_at_new_toplevel_block(cleaned)
    return sample.prompt + extracted_body