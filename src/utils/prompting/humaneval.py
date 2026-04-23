
# src/utils/prompting/humaneval.py

"""
HumanEval 전용 프롬프트 및 코드 추출 유틸리티
"""
from src.tasks.humaneval import HumanEvalSample
from .common import strip_code_fence, truncate_at_new_toplevel_block
import re


def build_humaneval_prompt(sample: HumanEvalSample) -> str:
    """
    HumanEval은 원본 prompt 자체가 함수 시그니처 + docstring completion 형식이라 그대로 사용
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
def build_humaneval_repair_prompt(
    sample: HumanEvalSample,
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
{sample.prompt}

[Previous Solution]
{previous_code}

[Error Message]
{error_message}

[Corrected Solution]
"""

def build_humaneval_refinement_prompt(
    sample: HumanEvalSample,
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
{sample.prompt}

[Previous Solution]
{previous_code}

[Improved Solution]
"""

def extract_humaneval_full_function_code(raw_output: str, entry_point: str, fallback_prompt: str) -> str:
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