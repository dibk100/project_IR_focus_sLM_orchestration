"""
Simple Repair Loop Orchestration

[code 생성] -> [execute (test)] -> [error message] -> [fix]

흐름:
1. task 읽기
2. initial generation
3. 실행/평가
4. 실패 시 adapter의 repair prompt로 재생성
5. 최대 max_repair_attempts까지 반복
6. attempt별 결과 저장 + task별 최종 결과 요약
"""
import os
import time
import yaml
from datetime import datetime

from src.models.hf_model import HFModel

from src.tasks.humaneval import HumanEvalTask
from src.tasks.mbpp import MBPPTask

from src.adapters.humaneval import HumanEvalAdapter
from src.adapters.mbpp import MBPPAdapter

from src.evaluation.metrics import (
    summarize_phase1_results,
    summarize_mbpp_failure_breakdown,
)

from src.utils.io import save_result, save_results_jsonl


def load_task_and_adapter(dataset_name: str):
    """
    dataset 이름에 따라 task loader와 adapter를 함께 반환한다.
    """
    if dataset_name == "humaneval":
        return HumanEvalTask(), HumanEvalAdapter()

    if dataset_name == "mbpp":
        return MBPPTask(), MBPPAdapter()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def run_repair_loop(config_path: str):
    """Simple repair loop 실험 실행"""
    # 1. Config 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    max_new_tokens = config["model"].get("max_new_tokens", 512)
    temperature = config["model"].get("temperature", 0.0)

    num_samples = config.get("num_samples", 1)
    output_dir = config.get("output_dir", "results/phase1_ver2/repair")
    dataset_name = config.get("dataset", "humaneval")
    method_name = config.get("method_name", "repair")
    max_repair_attempts = config.get("max_repair_attempts", 2)

    print("=" * 60)
    print("🔁 Simple Repair Loop 실험")
    print("=" * 60)

    # 2. Task / Adapter 로드
    task, adapter = load_task_and_adapter(dataset_name)
    print(f"📦 데이터셋: {dataset_name} | size={len(task)}")
    print(f"🔧 최대 repair 횟수: {max_repair_attempts}")

    # 3. 모델 로드
    print(f"🔄 모델 로딩: {model_name}")
    model = HFModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    print("✅ 모델 로딩 완료")

    samples_to_run = min(num_samples, len(task))

    # attempt-level 결과 저장용
    all_attempt_results = []

    # task-level 최종 metric 계산용
    final_eval_results = []

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        print(f"\n--- [{i + 1}/{samples_to_run}] {sample.task_id} ---")

        # initial prompt
        original_prompt = adapter.build_initial_prompt(sample)
        current_prompt = original_prompt

        previous_code = None
        final_exec_result = None

        for attempt_idx in range(max_repair_attempts + 1):
            # 4-1. generation
            gen_start = time.perf_counter()
            raw_output = model.generate(current_prompt)
            gen_end = time.perf_counter()
            latency_sec = gen_end - gen_start

            raw_text = raw_output

            # 4-2. code extraction
            generated_code = adapter.extract_code(sample, raw_text)

            # 4-3. execution
            exec_result = adapter.execute(sample, generated_code)
            final_exec_result = exec_result

            # 4-4. attempt record 생성
            attempt_record = adapter.make_attempt_record(
                sample=sample,
                method=method_name,
                model_name=model_name,
                attempt_idx=attempt_idx,
                prompt=current_prompt,
                raw_output=raw_text,
                generated_code=generated_code,
                latency_sec=latency_sec,
                exec_result=exec_result,
            )

            # 저장용 dict 변환
            result_entry = {
                "dataset": attempt_record.dataset,
                "task_id": attempt_record.task_id,
                "method": attempt_record.method,
                "model_name": attempt_record.model_name,
                "attempt_idx": attempt_record.attempt_idx,
                "prompt": attempt_record.prompt,
                "raw_output": attempt_record.raw_output,
                "generated_code": attempt_record.generated_code,
                "status": attempt_record.status,
                "passed": attempt_record.passed,
                "exec_success": attempt_record.exec_success,
                "error_type": attempt_record.error_type,
                "error_message": attempt_record.error_message,
                "latency_sec": attempt_record.latency_sec,
                "meta": attempt_record.meta,
            }

            if hasattr(sample, "entry_point"):
                result_entry["entry_point"] = sample.entry_point

            result_entry["timeout"] = attempt_record.meta.get("timeout", False)
            all_attempt_results.append(result_entry)

            pretty_status = {
                "pass": "✅ PASS",
                "fail": "❌ FAIL",
                "timeout": "⏱️ TIMEOUT",
            }.get(attempt_record.status, "❓ UNKNOWN")
            print(f"  attempt {attempt_idx}: {pretty_status}")

            previous_code = generated_code

            # 성공하면 종료
            if attempt_record.passed:
                break

            # 마지막 attempt면 종료
            if attempt_idx == max_repair_attempts:
                break

            # 4-5. 다음 repair prompt 생성 (adapter에 위임)
            current_prompt = adapter.build_repair_prompt(
                sample=sample,
                previous_code=previous_code,
                error_message=attempt_record.error_message,
            )

        # task-level 최종 결과 저장
        final_eval_results.append(final_exec_result)

    # 5. 결과 요약
    summary = summarize_phase1_results(final_eval_results)

    print(f"\n{'=' * 60}")
    print("📊 결과 요약 (task-level final result)")
    print(f"  총 문제: {summary['total']}")
    print(f"  통과: {summary['passed']}")
    print(f"  실행 성공: {summary['exec_success']}")
    print(f"  pass@1: {summary['pass@1']:.4f}")
    print(f"  execution_success_rate: {summary['execution_success_rate']:.4f}")
    print(f"  conditional_pass: {summary['conditional_pass']:.4f}")
    print(f"{'=' * 60}")

    extra_summary = {}
    if dataset_name == "mbpp":
        extra_summary = summarize_mbpp_failure_breakdown(final_eval_results)
        print("📌 MBPP Failure Breakdown (final attempt)")
        print(f"  code_failed: {extra_summary['code_failed']}")
        print(f"  setup_failed: {extra_summary['setup_failed']}")
        print(f"  test_failed: {extra_summary['test_failed']}")
        print(f"  semantic_failed: {extra_summary['semantic_failed']}")
        print(f"  execution_failed: {extra_summary['execution_failed']}")
        print(f"{'=' * 60}")

    # 6. 결과 저장
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
            "extra_summary": extra_summary,
        },
        os.path.join(output_dir, "summary.json"),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.repair <config.yaml>")
        sys.exit(1)

    run_repair_loop(sys.argv[1])