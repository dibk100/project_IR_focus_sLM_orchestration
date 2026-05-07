"""
Planner-Coder-Repair 전용 Prompt Builder

기존 repair prompt와의 차이:
- plan context (planner의 출력)를 함께 포함
- repairer가 원래 plan을 참고하되, plan이 틀렸다면 수정할 수 있도록 유도
"""

from src.tasks.humaneval import HumanEvalSample
from src.tasks.mbpp import MBPPSample


def build_humaneval_repair_with_plan_prompt(
    sample: HumanEvalSample,
    planner_output: str,
    previous_code: str,
    error_message: str,
) -> str:
    return f"""You are given a Python function task, a plan, a previous incorrect solution, and its execution error.

Your job is to repair the solution so that it passes the tests.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Do not include explanations.
- Keep the same function name and signature.
- Provide a complete corrected function.
- Use the plan as guidance, but fix the plan if it conflicts with the task or the error.

Task Prompt:
{sample.input}

Plan:
{planner_output}

Previous Solution:
{previous_code}

Error Message:
{error_message}

Corrected Solution:
"""


def build_mbpp_repair_with_plan_prompt(
    sample: MBPPSample,
    planner_output: str,
    previous_code: str,
    error_message: str,
) -> str:
    return f"""You are given a Python function task, a plan, a previous incorrect solution, and its execution error.

Your job is to repair the solution so that it passes the tests.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Do not include explanations.
- Keep the exact function name and signature implied by the test hint.
- Provide a complete corrected function.
- Use the plan as guidance, but fix the plan if it conflicts with the task, the test hint, or the error.

Task Prompt:
{sample.input}

Test hint:
{sample.hint}

Plan:
{planner_output}

Previous Solution:
{previous_code}

Error Message:
{error_message}

Corrected Solution:
"""