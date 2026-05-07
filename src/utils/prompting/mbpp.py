# src/utils/prompting/mbpp.py

"""
MBPP 전용 프롬프트 및 코드 추출 유틸리티
"""

import re

from src.tasks.mbpp import MBPPSample
from .common import strip_code_fence


def build_mbpp_prompt(sample: MBPPSample) -> str:
    return f"""Write Python code only.
Solve the problem below.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Do not include explanations.
- Use the exact function name and arguments implied by the test hint.
- Include any needed helper classes or functions.
- Provide a complete, executable solution.

Problem:
{sample.input}

Test hint:
{sample.hint}

Code:
"""


def extract_mbpp_code(raw_output: str) -> str:
    """
    MBPP 모델 출력에서 실행 가능한 코드만 추출한다.

    전략:
    1. fenced code block이 여러 개 있으면 마지막 block 사용
    2. fenced block이 없으면 def/class부터 시작하는 코드 추출
    3. 그래도 없으면 fence 제거 후 반환
    """
    text = raw_output.strip()

    fenced_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fenced_blocks:
        return fenced_blocks[-1].rstrip()

    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class "):
            start_idx = i
            break

    if start_idx is not None:
        return "\n".join(lines[start_idx:]).rstrip()

    return strip_code_fence(text)


def build_mbpp_repair_prompt(
    sample: MBPPSample,
    previous_code: str,
    error_message: str | None,
) -> str:
    error_message = error_message or "Unknown execution error."

    return f"""You are given a Python programming task, a previous incorrect solution, and its execution error.

Your job is to repair the solution so that it passes the tests.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Do not include explanations.
- Keep the exact function name and signature implied by the test hint.
- Provide a complete corrected solution.

Problem:
{sample.input}

Test hint:
{sample.hint}

Previous solution:
{previous_code}

Error message:
{error_message}

Corrected code:
"""


def build_mbpp_refinement_prompt(
    sample: MBPPSample,
    previous_code: str,
) -> str:
    return f"""You are given a Python programming task and a previous candidate solution.

Your job is to improve the solution so that it is more likely to be correct.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Do not include explanations.
- Keep the exact function name and signature implied by the test hint.
- Include any needed helper classes or functions.
- Improve the code if needed; otherwise return a clean complete solution.

Problem:
{sample.input}

Test hint:
{sample.hint}

Previous solution:
{previous_code}

Improved code:
"""