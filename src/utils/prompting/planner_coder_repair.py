"""
Planner-Coder-Repair 전용 Prompt Builder

기존 repair prompt와의 차이:
- plan context (planner의 출력)를 함께 포함
- repairer가 원래 plan에 따라 코드를 수정하도록 유도
"""
from src.tasks.humaneval import HumanEvalSample
from src.tasks.mbpp import MBPPSample


def build_humaneval_repair_with_plan_prompt(
    sample: HumanEvalSample,
    planner_output: str,
    previous_code: str,
    error_message: str,
) -> str:
    """
    HumanEval용 repair prompt (plan 포함).
    이전 코드와 에러 메시지, 그리고 원래 plan을 함께 제공.
    """
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


def build_mbpp_repair_with_plan_prompt(
    sample: MBPPSample,
    planner_output: str,
    previous_code: str,
    error_message: str,
) -> str:
    """
    MBPP용 repair prompt (plan 포함).
    이전 코드와 에러 메시지 제공.
    """
    test_hint = sample.test_list[0] if sample.test_list else ""

    return f"""You are given a Python function task, a previous incorrect solution, and its execution error.

Your job is to repair the solution so that it passes the tests.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Keep the same function name and signature.
- Provide a complete corrected function.

[Task Prompt]
{sample.problem_text}

[Test hint]
{test_hint}

[Previous Solution]
{previous_code}

[Error Message]
{error_message}

[Corrected Solution]
"""
