"""
Simple Repair Loop Orchestration

[code 생성] → [execute (test)] → [error message] → [fix]

흐름:
1. task 읽기
2. initial generation
3. 실행/평가
4. 실패 시 error message를 포함하여 repair prompt로 재생성
5. 최대 max_repair_attempts까지 반복
6. attempt별 결과 저장 + task별 최종 결과 요약
"""
import os
import time
import yaml
from datetime import datetime

from src.tasks.humaneval import HumanEvalTask
from src.models.hf_model import HFModel
from src.utils.prompting import build_prompt, extract_code
from src.evaluation.executor import execute_code
from src.evaluation.metrics import summarize_results
from src.utils.io import save_result, save_results_jsonl


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


def build_repair_prompt(
    original_prompt: str,
    previous_code: str,
    error_message: str | None,
) -> str:
    """실패한 코드와 에러 메시지를 바탕으로 repair prompt 생성."""
    error_message = error_message or "Unknown execution error."

    return f"""You are given a Python function task, a previous incorrect solution, and its execution error.

Your job is to fix the solution so that it passes the tests.

Requirements:
- Return only Python code.
- Do not include markdown fences.
- Keep the same function name and signature.
- Provide a complete corrected function.

[Task Prompt]
{original_prompt}

[Previous Solution]
{previous_code}

[Execution Error]
{error_message}

[Corrected Solution]
"""


def run_repair_loop(config_path: str):
    """Simple repair loop 실험 실행."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    max_new_tokens = config["model"].get("max_new_tokens", 512)
    temperature = config["model"].get("temperature", 0.0)

    num_samples = config.get("num_samples", 1)
    output_dir = config.get("output_dir", "results/phase1_easy/repair")
    dataset_name = config.get("dataset", "humaneval")
    max_repair_attempts = config.get("max_repair_attempts", 2)

    method_name = "repair"

    print("=" * 60)
    print("🔁 Simple Repair Loop 실험")
    print("=" * 60)

    task = HumanEvalTask()
    print(f"📦 데이터셋: {task}")
    print(f"🔧 최대 repair 횟수: {max_repair_attempts}")

    print(f"🔄 모델 로딩: {model_name}")
    model = HFModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    print("✅ 모델 로딩 완료")

    samples_to_run = min(num_samples, len(task))

    all_attempt_results = []
    final_task_results = []

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        print(f"\n--- [{i+1}/{samples_to_run}] {sample.task_id} ---")

        original_prompt = build_prompt(sample.prompt)

        current_prompt = original_prompt
        previous_code = None
        final_exec_result = None

        for attempt_idx in range(max_repair_attempts + 1):
            # generation
            gen_start = time.perf_counter()
            raw_output = model.generate(current_prompt)
            gen_end = time.perf_counter()
            latency_sec = gen_end - gen_start

            # code extraction
            code = extract_code(sample.prompt, raw_output, sample.entry_point)

            # execution
            exec_result = execute_code(code, sample.test, sample.entry_point)
            status = get_status(exec_result)
            error_type, error_message = classify_error(exec_result)

            pretty_status = {
                "pass": "✅ PASS",
                "fail": "❌ FAIL",
                "timeout": "⏱️ TIMEOUT",
            }[status]
            print(f"  attempt {attempt_idx}: {pretty_status}")

            attempt_entry = {
                "dataset": dataset_name,
                "task_id": sample.task_id,
                "entry_point": sample.entry_point,
                "method": method_name,
                "model_name": model_name,
                "attempt_idx": attempt_idx,
                "status": status,
                "passed": exec_result.passed,
                "timeout": exec_result.timeout,
                "raw_output": raw_output,
                "generated_code": code,
                "error_type": error_type,
                "error_message": error_message,
                "latency_sec": latency_sec,
            }
            all_attempt_results.append(attempt_entry)

            final_exec_result = exec_result
            previous_code = code

            # 성공하면 종료
            if exec_result.passed:
                break

            # 마지막 attempt면 종료
            if attempt_idx == max_repair_attempts:
                break

            # 다음 repair prompt 구성
            current_prompt = build_repair_prompt(
                original_prompt=original_prompt,
                previous_code=previous_code,
                error_message=error_message,
            )

        final_task_results.append(
            type(
                "R",
                (),
                {
                    "passed": final_exec_result.passed,
                    "timeout": final_exec_result.timeout,
                },
            )()
        )

    summary = summarize_results(final_task_results)

    print(f"\n{'=' * 60}")
    print("📊 결과 요약 (task-level final result)")
    print(f"  총 문제: {summary['total']}")
    print(f"  통과: {summary['passed']}")
    print(f"  실패: {summary['failed']}")
    print(f"  타임아웃: {summary['timed_out']}")
    print(f"  final success rate: {summary['pass@1']:.4f}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_results_jsonl(
        all_attempt_results,
        os.path.join(output_dir, "details.jsonl"),
    )

    save_result(
        {
            "experiment": {
                "phase": "phase1_easy",
                "orchestration": "simple-repair-loop",
                "task": dataset_name,
                "timestamp": timestamp,
                "config_path": config_path,
            },
            "model": {
                "name": model_name,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
            "repair": {
                "max_repair_attempts": max_repair_attempts,
            },
            "summary": summary,
        },
        os.path.join(output_dir, "summary.json"),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.repair <config.yaml>")
        sys.exit(1)

    run_repair_loop(sys.argv[1])