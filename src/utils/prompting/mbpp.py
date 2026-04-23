# src/utils/prompting/mbpp.py

"""
MBPP 전용 프롬프트 및 코드 추출 유틸리티
"""

from src.tasks.mbpp import MBPPSample
from .common import strip_code_fence
import re


def build_mbpp_prompt(sample: MBPPSample) -> str:
    test_hint = sample.test_list[0] if sample.test_list else ""

    return (
        "Write Python code only.\n"
        "Solve the problem below.\n"
        "Use the exact function name and arguments required by the test.\n"
        "Include any needed helper classes or functions.\n\n"
        f"Problem:\n{sample.problem_text}\n\n"
        f"Test hint:\n{test_hint}\n"
    )


def extract_mbpp_code(raw_output: str) -> str:
    """
    MBPP 모델 출력에서 실행 가능한 코드만 추출한다.

    전략:
    1. fenced code block이 여러 개 있으면 마지막 block 사용
    2. fenced block이 없으면 def/class부터 시작하는 코드 추출
    3. 그래도 없으면 fence 제거 후 반환
    """
    text = raw_output.strip()

    # 1) fenced code block이 여러 개 있으면 마지막 block 선택
    fenced_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fenced_blocks:
        return fenced_blocks[-1].rstrip()

    # 2) fenced block이 없으면 def/class부터 코드 시작점 탐색
    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class "):
            start_idx = i
            break

    if start_idx is not None:
        code_lines = lines[start_idx:]
        return "\n".join(code_lines).rstrip()

    # 3) 최후 fallback
    return strip_code_fence(text)

def build_mbpp_repair_prompt(
    sample: MBPPSample,
    previous_code: str,
    error_message: str | None,
) -> str:
    error_message = error_message or "Unknown execution error."
    test_hint = sample.test_list[0] if sample.test_list else ""

    return f"""You are given a Python function task, a previous incorrect solution, and its execution error.

Your job is to repair the solution so that it passes the tests.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Keep the same function name and signature.
- Provide a complete corrected function.

Problem:
{sample.problem_text}

Test hint:
{test_hint}

Previous solution:
{previous_code}

Error Message:
{error_message}

Corrected code:
"""

def build_mbpp_refinement_prompt(
    sample: MBPPSample,
    previous_code: str,
) -> str:
    test_hint = sample.test_list[0] if sample.test_list else ""

    return f"""You are given a Python programming task and a previous candidate solution.

Your job is to improve the solution so that it is more likely to be correct.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Do not include explanations.
- Keep the exact target function name and signature from the task.
- Improve the code if needed; otherwise return a clean complete solution.

Problem:
{sample.problem_text}

Test hint:
{test_hint}

Previous solution:
{previous_code}

Improved code:
"""