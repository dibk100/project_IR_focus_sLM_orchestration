"""
Single-Shot Orchestration
task 읽기 → 모델 호출 → 코드 추출 → 실행/평가 → 결과 저장

전체 파이프라인의 가장 기본적인 baseline.

역할 :
config 읽기
task / adapter 선택
model 호출
attempt record 저장
metrics / save

adapter로 넘긴 역할 :
prompt 생성
code extraction
execution
execution 결과 해석
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
    dataset 이름에 따라 task loader와 adapter를 함께 반환
    """
    if dataset_name == "humaneval":
        return HumanEvalTask(), HumanEvalAdapter()

    if dataset_name == "mbpp":
        return MBPPTask(), MBPPAdapter()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def normalize_model_output(raw_output) -> str:
    """
    LLM.generate() 반환 타입을 방어적으로 정리
    LM이 출력을 문자열(str)이 아닌 타입을 출력할 경우, str로 변환하는 방어
    
    - str 반환 시 그대로 사용
    - dict 반환 시 text 필드 사용
    """
    if isinstance(raw_output, dict):
        return raw_output.get("text", "")
    return raw_output


def run_single_shot(config_path: str):
    """Single-shot baseline 실험 실행"""
    # 1. Config 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    max_new_tokens = config["model"].get("max_new_tokens", 512)
    temperature = config["model"].get("temperature", 0.0)
    num_samples = config.get("num_samples", 1)
    output_dir = config.get("output_dir", "results/non/single")

    dataset_name = config.get("dataset", "non")
    method_name = config.get("method_name", "non")

    # 2. Task / Adapter 로드
    print("=" * 60)
    print("📋 Single-Shot Baseline 실험")
    print("=" * 60)

    task, adapter = load_task_and_adapter(dataset_name)
    print(f"📦 데이터셋: {dataset_name} | size={len(task)}")

    # 3. 모델 로드
    print(f"🔄 모델 로딩: {model_name}")
    model = HFModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    print("✅ 모델 로딩 완료")

    # 4. 실험 실행
    all_results = []
    eval_results = []

    samples_to_run = min(num_samples, len(task))

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        print(f"\n--- [{i + 1}/{samples_to_run}] {sample.task_id} ---")

        # 4-1. prompt 구성
        prompt = adapter.build_initial_prompt(sample)

        # 4-2. 모델 호출 + 시간 측정
        gen_start = time.perf_counter()
        raw_output = model.generate(prompt)
        gen_end = time.perf_counter()
        latency_sec = gen_end - gen_start

        #raw_text = normalize_model_output(raw_output)
        raw_text = raw_output

        # 4-3. 코드 추출
        generated_code = adapter.extract_code(sample, raw_text)

        # 4-4. 실행 / 평가
        exec_result = adapter.execute(sample, generated_code)
        eval_results.append(exec_result)

        # 4-5. attempt record 생성
        attempt_record = adapter.make_attempt_record(
            sample=sample,
            method=method_name,
            model_name=model_name,
            attempt_idx=0,
            prompt=prompt,
            raw_output=raw_text,
            generated_code=generated_code,
            latency_sec=latency_sec,
            exec_result=exec_result,
        )

        # 저장용 dict로 변환
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

        # HumanEval이면 entry_point를 보기 좋게 최상단에도 남김
        if hasattr(sample, "entry_point"):
            result_entry["entry_point"] = sample.entry_point

        # timeout은 현재 HumanEval 쪽 meta에 들어있으므로 꺼내서 같이 저장
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
                "orchestration": "single-shot",
                "task": dataset_name,
                "timestamp": timestamp,
                "config_path": config_path,
            },
            "model": {
                "name": model_name,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
            "summary": summary,
            "extra_summary": extra_summary,
        },
        os.path.join(output_dir, "summary.json"),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.single <config.yaml>")
        sys.exit(1)

    run_single_shot(sys.argv[1])