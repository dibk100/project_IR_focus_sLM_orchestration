# src/utils/prompting/bigcode.py

"""
BigCode 전용 프롬프트 및 코드 추출 유틸리티
"""
from src.tasks.bigcode import BigCodeSample
from .common import strip_code_fence, truncate_at_new_toplevel_block
import re


def build_bigcode_prompt(sample: BigCodeSample) -> str:
    """
    BigCode은 instruct_prompt에 자연어와 코드조각(import)가 있음
    """
    return sample.instruct_prompt



def extract_bigcode_code(raw_output: str) -> str:
    """
    BigCode 모델 출력에서 실행 가능한 코드만 추출한다.

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

def build_bigcode_repair_prompt(
    sample: BigCodeSample,
    previous_code: str,
    error_message: str | None,
) -> str:
    """
    실패한 코드와 에러 메시지를 바탕으로 repair prompt 생성
    """
    
    error_message = error_message or "Unknown execution error."

    return f"""You are given a Python function task, a previous incorrect solution, and its execution error.

Your job is to repair the solution so that it passes the tests.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Keep the same function name and signature.
- Provide a complete corrected function.

[Task Prompt]
{sample.instruct_prompt}

[Previous Solution]
{previous_code}

[Error Message]
{error_message}

[Corrected Solution]
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

[Task Prompt]
{sample.instruct_prompt}

[Previous Solution]
{previous_code}

[Improved Solution]
"""

def extract_bigcode_full_function_code(raw_output: str, entry_point: str, fallback_prompt: str) -> str:
    """planner/coder용: raw_output에서 target 함수 전체를 추출."""
    text = raw_output.strip()

    # markdown 제거
    text = text.replace("```python", "")
    text = text.replace("```", "").strip()

    lines = text.splitlines()

    imports = []
    function_block = []

    in_target_function = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # import 수집
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
            continue

        # target 함수 시작
        if re.match(rf"def\s+{re.escape(entry_point)}\s*\(", stripped):
            in_target_function = True

        if in_target_function:
            # 다른 함수 나오면 종료
            if (
                function_block
                and line.startswith("def ")
                and not re.match(rf"def\s+{re.escape(entry_point)}\s*\(", stripped)
            ):
                break

            function_block.append(line)

    # fallback
    if not function_block:
        # 붕괴 출력 방지: 너무 짧거나 code-like 하지 않으면 빈 코드로 반환
        stripped = text.strip()
        if len(stripped) < 20 or stripped in {"s", "p"}:
            return fallback_prompt
        return fallback_prompt + "\n" + stripped

    # 최종 조합: imports + function
    final_code = "\n".join(imports + [""] + function_block).rstrip() + "\n"
    return final_code