"""
프롬프트 구성 및 코드 추출 유틸리티
"""
import re


def build_prompt(task_prompt: str) -> str:
    """HumanEval 프롬프트를 모델 입력용으로 구성
    
    HumanEval의 prompt는 함수 시그니처 + docstring으로 구성되어 있어
    모델이 함수 본문을 완성하도록 그대로 전달한다.
    """
    return task_prompt


def extract_code(prompt: str, generation: str, entry_point: str) -> str:
    """모델 생성 결과에서 함수 본문 코드를 추출
    
    Args:
        prompt: 원본 프롬프트 (함수 시그니처 + docstring)
        generation: 모델 생성 텍스트
        entry_point: 함수 이름
    
    Returns:
        prompt + 추출된 함수 본문 (실행 가능한 완전한 함수)
    """
    # markdown code fence 최소 제거
    generation = generation.replace("```python", "")
    generation = generation.replace("```", "")
    generation = generation.strip("\n")
    
    # 생성된 텍스트에서 함수 본문 부분만 추출
    # 다음 함수 정의나 클래스 정의가 나오면 거기서 끊음
    lines = generation.split("\n")
    code_lines = []
    for line in lines:
        # 새로운 top-level 정의가 나오면 중단
        if code_lines and (
            line.startswith("def ") 
            or line.startswith("class ")
            or line.startswith("if __name__")
        ):
            break
        code_lines.append(line)

    extracted = "\n".join(code_lines)
    
    # prompt + 생성된 본문을 합쳐서 완전한 함수 코드를 만듦
    full_code = prompt + extracted
    return full_code
