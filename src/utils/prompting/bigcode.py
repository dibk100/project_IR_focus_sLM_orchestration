# src/utils/prompting/bigcode.py

"""
BigCode 전용 프롬프트 및 코드 추출 유틸리티
"""

import re

from src.tasks.bigcode import BigCodeSample
from .common import strip_code_fence


def build_bigcode_prompt(sample: BigCodeSample) -> str:
    """
    BigCodeBench는 instruct_prompt에 자연어 설명과 코드 scaffold가 포함됨.
    """
    return sample.input


def extract_bigcode_code(raw_output: str) -> str:
    """
    BigCode 모델 출력에서 실행 가능한 코드만 추출한다.

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


def build_bigcode_repair_prompt(
    sample: BigCodeSample,
    previous_code: str,
    error_message: str | None,
) -> str:
    error_message = error_message or "Unknown execution error."

    return f"""You are given a Python function task, a previous incorrect solution, and its execution error.

Your job is to repair the solution so that it passes the tests.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Do not include explanations.
- Keep the same function name and signature.
- Provide a complete corrected function.

Task:
{sample.input}

Previous solution:
{previous_code}

Error message:
{error_message}

Corrected code:
"""


def build_bigcode_refinement_prompt(
    sample: BigCodeSample,
    previous_code: str,
) -> str:
    return f"""You are given a Python programming task and a previous candidate solution.

Your job is to improve the solution so that it is more likely to be correct.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Do not include explanations.
- Keep the exact target function name and signature from the task.
- Improve the code if needed; otherwise return a clean complete solution.

Task:
{sample.input}

Previous solution:
{previous_code}

Improved code:
"""


def extract_bigcode_full_function_code(
    raw_output: str,
    entry_point: str,
    fallback_prompt: str,
) -> str:
    """planner/coder용: raw_output에서 target 함수 전체를 추출."""
    text = raw_output.strip()

    text = text.replace("```python", "")
    text = text.replace("```", "").strip()

    lines = text.splitlines()

    imports = []
    function_block = []
    in_target_function = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
            continue

        if re.match(rf"def\s+{re.escape(entry_point)}\s*\(", stripped):
            in_target_function = True

        if in_target_function:
            if (
                function_block
                and line.startswith("def ")
                and not re.match(rf"def\s+{re.escape(entry_point)}\s*\(", stripped)
            ):
                break

            function_block.append(line)

    if not function_block:
        stripped = text.strip()
        if len(stripped) < 20 or stripped in {"s", "p"}:
            return fallback_prompt
        return fallback_prompt + "\n" + stripped

    return "\n".join(imports + [""] + function_block).rstrip() + "\n"