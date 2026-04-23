from src.tasks.humaneval import HumanEvalSample
from src.tasks.mbpp import MBPPSample
from src.tasks.bigcode import BigCodeSample

from src.utils.prompting.planner_coder import *
from src.utils.prompting.humaneval import *
from src.utils.prompting.mbpp import *
from src.utils.prompting.bigcode import *

def build_planner_prompt_for_sample(sample) -> str:
    """
    sample 타입에 따라 planner prompt를 선택한다.
    """
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_planner_prompt(sample)

    if isinstance(sample, MBPPSample):
        return build_mbpp_planner_prompt(sample)

    if isinstance(sample, BigCodeSample):
        return build_bigcode_planner_prompt(sample)

    raise TypeError(f"Unsupported sample type: {type(sample)}")


def build_coder_prompt_for_sample(sample, planner_output: str) -> str:
    """
    sample 타입에 따라 coder prompt를 선택한다.
    """
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_coder_prompt(sample, planner_output)

    if isinstance(sample, MBPPSample):
        return build_mbpp_coder_prompt(sample, planner_output)

    if isinstance(sample, BigCodeSample):
        return build_bigcode_coder_prompt(sample, planner_output)

    raise TypeError(f"Unsupported sample type: {type(sample)}")

def build_repair_prompt_for_sample(
    sample,
    previous_code: str | None,
    error_message: str | None,
    failing_status: str | None,
    planner_output: str | None = None,   # <- 나중 3번 방식 대비
):
    """
    1차 버전:
    - planner_output 없이도 동작
    - 나중에 planner_output을 prompt에 넣는 3번 방식으로 확장 가능
    """

    problem_text = getattr(sample, "prompt", None) or getattr(sample, "question", None) or str(sample)

    entry_point_text = ""
    if hasattr(sample, "entry_point"):
        entry_point_text = f"\nEntry point: {sample.entry_point}\n"

    planner_block = ""
    if planner_output:
        planner_block = f"\nImplementation plan:\n{planner_output}\n"

    return f"""You are repairing a previously generated Python Code.

Problem:
{problem_text}
{entry_point_text}

Previous incorrect code:
```python
{previous_code or ""}
```
Observed failure status:
{failing_status}

Observed Error message:
{error_message or "N/A"}

{planner_block}
Please repair the code so that it correctly solves the problem and passes the tests.

Return only the final corrected Python code:
"""

def build_replanner_prompt_for_sample(
    sample,
    previous_plan: str,
    failing_status: str,
    error_type: str | None,
    error_message: str | None,
    previous_code: str | None = None,
):
    """
    이전 plan과 최근 실패 정보를 반영해 수정된 plan을 생성하도록 유도하는 replanner prompt.
    """

    previous_code_block = previous_code if previous_code else "(omitted)"

    return f"""You are revising a Python solution plan after a failed attempt.

Task:
{sample.prompt}

Previous plan:
{previous_plan}

Latest failure:
- status: {failing_status}
- error_type: {error_type or "NONE"}
- error_message: {error_message or "NONE"}

Previous code:
{previous_code_block}

Write a revised short plan that fixes the likely cause of the failure.

Rules:
- Do NOT write code.
- Use at most 3 bullet points.
- Keep the original task requirements unchanged.
- Revise the previous plan instead of repeating it unchanged.
- Focus on the likely cause of failure.
- Do NOT introduce unnecessary helper functions.
- Do NOT add assumptions not stated in the task.
- Focus only on the core algorithm and correction.

Revised Plan:
"""