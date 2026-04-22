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

Write a very short plan.

Rules:
- Do NOT write code.
- Use at most 3 bullet points.
- Do NOT introduce helper functions unless absolutely necessary.
- Do NOT introduce new variable names beyond those implied by the task.
- Do NOT add assumptions not stated in the task.
- Focus only on the core algorithm.

You must respect the function interface implied by this test:
{test_hint}

Plan:
"""

def build_mbpp_coder_prompt(sample: MBPPSample, planner_output: str) -> str:
    test_hint = sample.test_list[0] if sample.test_list else ""
    return f"""You are writing Python code.

Task:
{sample.problem_text}

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

Test hint:
{test_hint}

Code:
"""

def extract_planner_output(raw_output: str) -> str:
    return raw_output.strip()


def build_bigcode_planner_prompt(sample: BigCodeSample) -> str:
    """문제 해결 계획만 생성하도록 유도하는 planner prompt."""
    return f"""You are planning a Python solution.

Task:
{sample.instruct_prompt}

Write a concise plan to solve the problem.

Rules:
- Do NOT write code
- Use 3–5 bullet points
- Describe the key algorithm steps clearly
- You may mention variables and operations if helpful
- Focus on correctness

Plan:
"""

def build_bigcode_coder_prompt(sample: BigCodeSample, planner_output: str) -> str:
    """planner의 계획을 바탕으로 코드만 생성하도록 유도하는 coder prompt."""
    return f"""You are writing Python code.

Task:
{sample.instruct_prompt}

Plan:
{planner_output}

Instructions:
- Write correct Python code that solves the problem
- The code must be complete and runnable
- Follow the plan, but fix it if necessary
- Keep the exact function name and signature

Rules:
- Output only Python code
- Do not include explanations or markdown
- Do not output trivial code (e.g., pass, single variables)
- Define all variables before use
- Include imports if needed

Code:
"""