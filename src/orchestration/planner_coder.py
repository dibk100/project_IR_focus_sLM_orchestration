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

from src.models.hf_model import HFModel

from src.tasks.humaneval import HumanEvalTask
from src.tasks.mbpp import MBPPTask
from src.tasks.humaneval import HumanEvalSample
from src.tasks.mbpp import MBPPSample

from src.adapters.humaneval import HumanEvalAdapter
from src.adapters.mbpp import MBPPAdapter

from src.evaluation.metrics import (
    summarize_phase1_results,
    summarize_mbpp_failure_breakdown,
)

from src.utils.io import save_result, save_results_jsonl
from src.utils.prompting.planner_coder import (
    build_humaneval_planner_prompt,
    build_mbpp_planner_prompt,
    build_humaneval_coder_prompt,
    build_mbpp_coder_prompt,
    extract_planner_output,
)


def load_task_and_adapter(dataset_name: str):
    """
    dataset 이름에 따라 task loader와 adapter를 함께 반환한다.
    """
    if dataset_name == "humaneval":
        return HumanEvalTask(), HumanEvalAdapter()

    if dataset_name == "mbpp":
        return MBPPTask(), MBPPAdapter()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_planner_prompt_for_sample(sample) -> str:
    """
    sample 타입에 따라 planner prompt를 선택한다.
    """
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_planner_prompt(sample)

    if isinstance(sample, MBPPSample):
        return build_mbpp_planner_prompt(sample)

    raise TypeError(f"Unsupported sample type: {type(sample)}")


def build_coder_prompt_for_sample(sample, planner_output: str) -> str:
    """
    sample 타입에 따라 coder prompt를 선택한다.
    """
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_coder_prompt(sample, planner_output)

    if isinstance(sample, MBPPSample):
        return build_mbpp_coder_prompt(sample, planner_output)

    raise TypeError(f"Unsupported sample type: {type(sample)}")


def run_planner_coder(config_path: str):
    """planner-coder baseline 실험 실행"""
    # 1. Config 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    planner_model_name = config["planner_model"]["name"]
    planner_max_new_tokens = config["planner_model"].get("max_new_tokens", 256)
    planner_temperature = config["planner_model"].get("temperature", 0.0)

    coder_model_name = config["coder_model"]["name"]
    coder_max_new_tokens = config["coder_model"].get("max_new_tokens", 512)
    coder_temperature = config["coder_model"].get("temperature", 0.0)

    num_samples = config.get("num_samples", 1)
    output_dir = config.get("output_dir", "results/phase1_ver2/planner_coder")
    dataset_name = config.get("dataset", "humaneval")
    method_name = config.get("method_name", "planner_coder")

    print("=" * 60)
    print("🧠 Planner-Coder 실험")
    print("=" * 60)

    # 2. Task / Adapter 로드
    task, adapter = load_task_and_adapter(dataset_name)
    print(f"📦 데이터셋: {dataset_name} | size={len(task)}")

    # 3. Planner 모델 로드
    print(f"🔄 Planner 모델 로딩: {planner_model_name}")
    planner_model = HFModel(
        model_name=planner_model_name,
        max_new_tokens=planner_max_new_tokens,
        temperature=planner_temperature,
    )
    print("✅ Planner 모델 로딩 완료")

    # 4. Coder 모델 로드
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
    eval_results = []

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        print(f"\n--- [{i + 1}/{samples_to_run}] {sample.task_id} ---")

        # 4-1. planner step
        planner_prompt = build_planner_prompt_for_sample(sample)

        planner_start = time.perf_counter()
        planner_raw_output = planner_model.generate(planner_prompt)
        planner_end = time.perf_counter()
        planner_latency_sec = planner_end - planner_start

        planner_output = extract_planner_output(planner_raw_output)

        # 4-2. coder step
        coder_prompt = build_coder_prompt_for_sample(sample, planner_output)

        coder_start = time.perf_counter()
        coder_raw_output = coder_model.generate(coder_prompt)
        coder_end = time.perf_counter()
        coder_latency_sec = coder_end - coder_start

        # 4-3. code extraction
        generated_code = adapter.extract_code_for_planner(
            sample,
            coder_raw_output,
        )

        # 4-4. execution / evaluation
        exec_result = adapter.execute(sample, generated_code)
        eval_results.append(exec_result)

        # 4-5. attempt record 생성
        total_latency_sec = planner_latency_sec + coder_latency_sec
        attempt_record = adapter.make_attempt_record(
            sample=sample,
            method=method_name,
            model_name=f"{planner_model_name} -> {coder_model_name}",
            attempt_idx=0,
            prompt=coder_prompt,
            raw_output=coder_raw_output,
            generated_code=generated_code,
            latency_sec=total_latency_sec,
            exec_result=exec_result,
        )

        # 저장용 dict 변환
        result_entry = {
            "dataset": attempt_record.dataset,
            "task_id": attempt_record.task_id,
            "method": attempt_record.method,
            "planner_model_name": planner_model_name,
            "coder_model_name": coder_model_name,
            "attempt_idx": attempt_record.attempt_idx,
            "status": attempt_record.status,
            "passed": attempt_record.passed,
            "exec_success": attempt_record.exec_success,
            "error_type": attempt_record.error_type,
            "error_message": attempt_record.error_message,
            "planner_prompt": planner_prompt,
            "planner_output": planner_output,
            "coder_prompt": coder_prompt,
            "raw_output": coder_raw_output,
            "generated_code": generated_code,
            "planner_latency_sec": planner_latency_sec,
            "coder_latency_sec": coder_latency_sec,
            "latency_sec": total_latency_sec,
            "meta": attempt_record.meta,
        }

        if hasattr(sample, "entry_point"):
            result_entry["entry_point"] = sample.entry_point

        result_entry["timeout"] = attempt_record.meta.get("timeout", False)
        all_results.append(result_entry)

        pretty_status = {
            "pass": "✅ PASS",
            "fail": "❌ FAIL",
            "timeout": "⏱️ TIMEOUT",
        }.get(attempt_record.status, "❓ UNKNOWN")
        print(f"  {pretty_status}")

    # 5. 결과 요약
    summary = summarize_phase1_results(eval_results)

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
        extra_summary = summarize_mbpp_failure_breakdown(eval_results)
        print("📌 MBPP Failure Breakdown")
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
            "extra_summary": extra_summary,
        },
        os.path.join(output_dir, "summary.json"),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.planner_coder <config.yaml>")
        sys.exit(1)

    run_planner_coder(sys.argv[1])