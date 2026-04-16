from src.tasks.humaneval import HumanEvalSample
from src.tasks.mbpp import MBPPSample
from src.tasks.bigcode import BigCodeSample

def build_humaneval_planner_prompt(sample: HumanEvalSample) -> str:
    """문제 해결 계획만 생성하도록 유도하는 planner prompt."""
    return f"""You are planning a Python solution.

Task:
{sample.prompt}

Write a very short plan.

Rules:
- Do NOT write code.
- Use at most 3 bullet points.
- Do NOT introduce helper functions unless absolutely necessary.
- Do NOT introduce new variable names beyond those implied by the task.
- Do NOT add assumptions not stated in the task.
- Focus only on the core algorithm.

Plan:
"""

def build_humaneval_coder_prompt(sample: HumanEvalSample, planner_output: str) -> str:
    """planner의 계획을 바탕으로 코드만 생성하도록 유도하는 coder prompt."""
    return f"""You are writing Python code.

Task:
{sample.prompt}

Plan:
{planner_output}

Write only the final Python code.

Rules:
- Output only code.
- Do not include markdown fences.
- Do not include explanations.
- Do not include any stray text such as single letters or comments outside the code.
- Keep the exact target function name and signature from the task.
- Do not introduce helper functions unless they are fully defined in the output.
- Do not use variables that are not defined in the function body or function signature.
- If imports are needed, include them explicitly.

Code:
"""

def build_mbpp_planner_prompt(sample: MBPPSample) -> str:
    test_hint = sample.test_list[0] if sample.test_list else ""
    return f"""You are planning a Python solution.

Task:
{sample.problem_text}

You must respect the function interface implied by this test:
{test_hint}

Write a very short plan.

Rules:
- Do NOT write code.
- Use at most 3 bullet points.
- Focus only on the core algorithm.
- Keep the expected function interface in mind.

Plan:
"""

def build_mbpp_coder_prompt(sample: MBPPSample, planner_output: str) -> str:
    test_hint = sample.test_list[0] if sample.test_list else ""
    return (
        "Write Python code only.\n"
        "Solve the problem below using the plan.\n"
        "Use the exact function name and arguments required by the test.\n"
        "Include any needed helper classes or functions.\n\n"
        f"Problem:\n{sample.problem_text}\n\n"
        f"Test hint:\n{test_hint}\n\n"
        f"Plan:\n{planner_output}\n"
    )

def extract_planner_output(raw_output: str) -> str:
    return raw_output.strip()

# ver1 - mpbb전용으로 새로 했으나 이게 맞나 고민
# def build_humaneval_coder_prompt(sample: HumanEvalSample, planner_output: str) -> str:
#     """planner의 계획을 바탕으로 코드만 생성하도록 유도하는 coder prompt."""
#     return f"""You are writing Python code.

# Task:
# {sample.prompt}

# Plan:
# {planner_output}

# Write only the final Python code.

# Rules:
# - Output only code.
# - Do not include markdown fences.
# - Do not include explanations.
# - Do not include any stray text such as single letters or comments outside the code.
# - Keep the exact target function name and signature from the task.
# - Do not introduce helper functions unless they are fully defined in the output.
# - Do not use variables that are not defined in the function body or function signature.
# - If imports are needed, include them explicitly.

# Code:
# """

# def build_mbpp_planner_prompt(sample: MBPPSample) -> str:
#     test_hint = sample.test_list[0] if sample.test_list else ""
#     return f"""You are planning a Python solution.

# Task:
# {sample.problem_text}

# You must respect the function interface implied by this test:
# {test_hint}

# Write a very short plan.

# Rules:
# - Do NOT write code.
# - Use at most 3 bullet points.
# - Focus only on the core algorithm.
# - Keep the expected function interface in mind.

# Plan:
# """

# def build_mbpp_coder_prompt(sample: MBPPSample, planner_output: str) -> str:
#     test_hint = sample.test_list[0] if sample.test_list else ""
#     return (
#         "Write Python code only.\n"
#         "Solve the problem below using the plan.\n"
#         "Use the exact function name and arguments required by the test.\n"
#         "Include any needed helper classes or functions.\n\n"
#         f"Problem:\n{sample.problem_text}\n\n"
#         f"Test hint:\n{test_hint}\n\n"
#         f"Plan:\n{planner_output}\n"
#     )


def build_bigcode_planner_prompt(sample: BigCodeSample) -> str:
    """문제 해결 계획만 생성하도록 유도하는 planner prompt."""
    return f"""You are planning a Python solution.

Task:
{sample.instruct_prompt}

Write a very short plan.

Rules:
- Do NOT write code.
- Use at most 3 bullet points.
- Do NOT introduce helper functions unless absolutely necessary.
- Do NOT introduce new variable names beyond those implied by the task.
- Do NOT add assumptions not stated in the task.
- Focus only on the core algorithm.

Plan:
"""

def build_bigcode_coder_prompt(sample: BigCodeSample, planner_output: str) -> str:
    """planner의 계획을 바탕으로 코드만 생성하도록 유도하는 coder prompt."""
    return f"""You are writing Python code.

Task:
{sample.instruct_prompt}

Plan:
{planner_output}

Write only the final Python code.

Rules:
- Output only code.
- Do not include markdown fences.
- Do not include explanations.
- Do not include any stray text such as single letters or comments outside the code.
- Keep the exact target function name and signature from the task.
- Do not introduce helper functions unless they are fully defined in the output.
- Do not use variables that are not defined in the function body or function signature.
- If imports are needed, include them explicitly.

Code:
"""