"""
Planner-Coder Orchestration

흐름:
1. task 읽기
2. planner가 구현 계획 생성
3. coder가 계획을 바탕으로 코드 생성
4. 실행/평가
5. 결과 저장

Phase 1에서는 repair 없이 planner/coder의 순수 효과를 본다.
"""
import os
import time
import yaml
from datetime import datetime

from src.tasks.humaneval import HumanEvalTask
from src.models.hf_model import HFModel
from src.utils.prompting import extract_code
from src.evaluation.executor import execute_code
from src.evaluation.metrics import summarize_results
from src.utils.io import save_result, save_results_jsonl

import re

def extract_full_function_code(raw_output: str, entry_point: str, fallback_prompt: str) -> str:
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
        return fallback_prompt + text

    # 최종 조합: imports + function
    final_code = "\n".join(imports + [""] + function_block).rstrip() + "\n"
    return final_code

def classify_error(exec_result) -> tuple[str | None, str | None]:
    """실행 결과를 바탕으로 에러 타입/메시지 정리."""
    if exec_result.passed:
        return None, None

    if exec_result.timeout:
        return "timeout", exec_result.error

    error_message = exec_result.error
    if error_message is None:
        return "unknown_error", None

    lowered = error_message.lower()
    if "syntaxerror" in lowered:
        return "syntax_error", error_message
    if "assertionerror" in lowered:
        return "assertion_error", error_message
    if "typeerror" in lowered:
        return "type_error", error_message
    if "nameerror" in lowered:
        return "name_error", error_message
    if "indexerror" in lowered:
        return "index_error", error_message
    if "keyerror" in lowered:
        return "key_error", error_message
    if "valueerror" in lowered:
        return "value_error", error_message
    if "attributeerror" in lowered:
        return "attribute_error", error_message
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return "import_error", error_message

    return "runtime_error", error_message


def get_status(exec_result) -> str:
    if exec_result.passed:
        return "pass"
    if exec_result.timeout:
        return "timeout"
    return "fail"


def build_planner_prompt(task_prompt: str) -> str:
    """문제 해결 계획만 생성하도록 유도하는 planner prompt."""
    return f"""You are a planning assistant for Python programming problems.

Given the task prompt, write a short implementation plan.

Requirements:
- Do NOT write code.
- Explain the intended algorithm briefly.
- Mention important edge cases or constraints.
- Keep it concise and structured.

[Task Prompt]
{task_prompt}

[Implementation Plan]
"""


def build_coder_prompt(task_prompt: str, planner_output: str) -> str:
    """planner의 계획을 바탕으로 코드만 생성하도록 유도하는 coder prompt."""
    return f"""You are a Python coding assistant.

Write a complete Python solution for the following task.

Requirements:
- Use the task prompt and the implementation plan.
- Return only Python code.
- Do not include markdown fences.
- Do not include explanation.
- Keep the exact function name and signature from the task prompt.

[Task Prompt]
{task_prompt}

[Implementation Plan]
{planner_output}

[Code]
"""


def run_planner_coder(config_path: str):
    """planner-coder baseline 실험 실행."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    planner_model_name = config["planner_model"]["name"]
    planner_max_new_tokens = config["planner_model"].get("max_new_tokens", 256)
    planner_temperature = config["planner_model"].get("temperature", 0.0)

    coder_model_name = config["coder_model"]["name"]
    coder_max_new_tokens = config["coder_model"].get("max_new_tokens", 512)
    coder_temperature = config["coder_model"].get("temperature", 0.0)

    num_samples = config.get("num_samples", 1)
    output_dir = config.get("output_dir", "results/phase1_easy/planner_coder")
    dataset_name = config.get("dataset", "humaneval")
    method_name = "planner_coder"

    print("=" * 60)
    print("🧠 Planner-Coder 실험")
    print("=" * 60)

    task = HumanEvalTask()
    print(f"📦 데이터셋: {task}")

    print(f"🔄 Planner 모델 로딩: {planner_model_name}")
    planner_model = HFModel(
        model_name=planner_model_name,
        max_new_tokens=planner_max_new_tokens,
        temperature=planner_temperature,
    )
    print("✅ Planner 모델 로딩 완료")

    if coder_model_name == planner_model_name:
        coder_model = planner_model
        print(f"🔁 Coder 모델은 Planner와 동일 모델 공유: {coder_model_name}")
    else:
        print(f"🔄 Coder 모델 로딩: {coder_model_name}")
        coder_model = HFModel(
            model_name=coder_model_name,
            max_new_tokens=coder_max_new_tokens,
            temperature=coder_temperature,
        )
        print("✅ Coder 모델 로딩 완료")

    samples_to_run = min(num_samples, len(task))
    all_results = []

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        print(f"\n--- [{i+1}/{samples_to_run}] {sample.task_id} ---")

        # 1) planner step
        planner_prompt = build_planner_prompt(sample.prompt)
        planner_start = time.perf_counter()
        planner_output = planner_model.generate(planner_prompt)
        planner_end = time.perf_counter()
        planner_latency_sec = planner_end - planner_start

        # 2) coder step
        coder_prompt = build_coder_prompt(sample.prompt, planner_output)
        coder_start = time.perf_counter()
        raw_output = coder_model.generate(coder_prompt)
        coder_end = time.perf_counter()
        coder_latency_sec = coder_end - coder_start

        # 3) code extraction
        code = extract_full_function_code(
            raw_output=raw_output,
            entry_point=sample.entry_point,
            fallback_prompt=sample.prompt,
        )

        # 4) execution / evaluation
        exec_result = execute_code(code, sample.test, sample.entry_point)
        status = get_status(exec_result)
        error_type, error_message = classify_error(exec_result)

        pretty_status = {
            "pass": "✅ PASS",
            "fail": "❌ FAIL",
            "timeout": "⏱️ TIMEOUT",
        }[status]
        print(f"  {pretty_status}")

        # 5) result record
        result_entry = {
            "dataset": dataset_name,
            "task_id": sample.task_id,
            "entry_point": sample.entry_point,
            "method": method_name,
            "planner_model_name": planner_model_name,
            "coder_model_name": coder_model_name,
            "status": status,
            "passed": exec_result.passed,
            "timeout": exec_result.timeout,
            "planner_prompt": planner_prompt,
            "planner_output": planner_output,
            "coder_prompt": coder_prompt,
            "raw_output": raw_output,
            "generated_code": code,
            "error_type": error_type,
            "error_message": error_message,
            "planner_latency_sec": planner_latency_sec,
            "coder_latency_sec": coder_latency_sec,
            "latency_sec": planner_latency_sec + coder_latency_sec,
        }
        all_results.append(result_entry)

    exec_results = [
        type("R", (), {"passed": r["passed"], "timeout": r["timeout"]})()
        for r in all_results
    ]
    summary = summarize_results(exec_results)

    print(f"\n{'=' * 60}")
    print("📊 결과 요약")
    print(f"  총 문제: {summary['total']}")
    print(f"  통과: {summary['passed']}")
    print(f"  실패: {summary['failed']}")
    print(f"  타임아웃: {summary['timed_out']}")
    print(f"  pass@1: {summary['pass@1']:.4f}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_results_jsonl(
        all_results,
        os.path.join(output_dir, "details.jsonl"),
    )

    save_result(
        {
            "experiment": {
                "phase": "phase1_easy",
                "orchestration": "planner-coder",
                "task": dataset_name,
                "timestamp": timestamp,
                "config_path": config_path,
            },
            "planner_model": {
                "name": planner_model_name,
                "max_new_tokens": planner_max_new_tokens,
                "temperature": planner_temperature,
            },
            "coder_model": {
                "name": coder_model_name,
                "max_new_tokens": coder_max_new_tokens,
                "temperature": coder_temperature,
            },
            "summary": summary,
        },
        os.path.join(output_dir, "summary.json"),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.planner_coder <config.yaml>")
        sys.exit(1)

    run_planner_coder(sys.argv[1])