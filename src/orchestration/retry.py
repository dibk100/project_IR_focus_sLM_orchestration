"""
Retry (Refinement-only) Orchestration

흐름:
1. task 읽기
2. initial generation
3. 이전 코드를 기반으로 refinement (error message 없이)
4. 실행/평가
5. 반복

repair :
1. 코드 생성
2. 실행 → 실패 + 에러 메시지
3. (문제 + 코드 + 에러) → 수정
4. 반복

핵심:
- execution feedback(에러 메시지)를 사용하지 않음
- pure refinement 효과 측정
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
    if dataset_name == "humaneval":
        return HumanEvalTask(), HumanEvalAdapter()

    if dataset_name == "mbpp":
        return MBPPTask(), MBPPAdapter()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def run_retry(config_path: str):
    """Retry (refinement-only) baseline 실행"""
    # 1. config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    max_new_tokens = config["model"].get("max_new_tokens", 512)
    temperature = config["model"].get("temperature", 0.0)

    num_samples = config.get("num_samples", 1)
    output_dir = config.get("output_dir", "results/phase1_ver2/retry")
    dataset_name = config.get("dataset", "humaneval")
    method_name = config.get("method_name", "retry")
    max_retry_attempts = config.get("max_retry_attempts", 1)

    print("=" * 60)
    print("🔁 Retry (Refinement-only) 실험")
    print("=" * 60)

    # 2. task / adapter
    task, adapter = load_task_and_adapter(dataset_name)
    print(f"📦 데이터셋: {dataset_name} | size={len(task)}")
    print(f"🔧 최대 retry 횟수: {max_retry_attempts}")

    # 3. model
    print(f"🔄 모델 로딩: {model_name}")
    model = HFModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    print("✅ 모델 로딩 완료")

    samples_to_run = min(num_samples, len(task))

    all_attempt_results = []
    final_eval_results = []

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        print(f"\n--- [{i+1}/{samples_to_run}] {sample.task_id} ---")

        original_prompt = adapter.build_initial_prompt(sample)
        current_prompt = original_prompt

        previous_code = None
        final_exec_result = None

        for attempt_idx in range(max_retry_attempts + 1):
            # generation
            gen_start = time.perf_counter()
            raw_output = model.generate(current_prompt)
            gen_end = time.perf_counter()
            latency_sec = gen_end - gen_start

            raw_text = raw_output

            # extraction
            generated_code = adapter.extract_code(sample, raw_text)

            # execution
            exec_result = adapter.execute(sample, generated_code)
            final_exec_result = exec_result

            # record
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

            # 마지막이면 종료
            if attempt_idx == max_retry_attempts:
                break

            # error 없이 refinement
            current_prompt = adapter.build_refinement_prompt(
                sample=sample,
                previous_code=previous_code,
            )

        final_eval_results.append(final_exec_result)

    # 4. summary
    summary = summarize_phase1_results(final_eval_results)

    print(f"\n{'=' * 60}")
    print("📊 결과 요약")
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

        print("📌 MBPP Failure Breakdown")
        print(f"  code_failed: {extra_summary['code_failed']}")
        print(f"  setup_failed: {extra_summary['setup_failed']}")
        print(f"  test_failed: {extra_summary['test_failed']}")
        print(f"  semantic_failed: {extra_summary['semantic_failed']}")
        print(f"  execution_failed: {extra_summary['execution_failed']}")
        print(f"{'=' * 60}")

    # 5. save
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
                "orchestration": "retry",
                "task": dataset_name,
                "timestamp": timestamp,
                "config_path": config_path,
            },
            "model": {
                "name": model_name,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
            "retry": {
                "max_retry_attempts": max_retry_attempts,
            },
            "summary": summary,
            "extra_summary": extra_summary,
        },
        os.path.join(output_dir, "summary.json"),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.retry <config.yaml>")
        sys.exit(1)

    run_retry(sys.argv[1])