"""
Single-Shot Orchestration
task 읽기 → 모델 호출 → 코드 추출 → 실행/평가 → 결과 저장

전체 파이프라인의 가장 기본적인 baseline.
"""
import os
import yaml
import time
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
    """pass/fail/timeout 상태 문자열 반환."""
    if exec_result.passed:
        return "pass"
    if exec_result.timeout:
        return "timeout"
    return "fail"


def run_single_shot(config_path: str):
    """Single-shot baseline 실험 실행

    Args:
        config_path: YAML 설정 파일 경로
    """
    # 1. Config 로드
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    max_new_tokens = config["model"].get("max_new_tokens", 512)
    temperature = config["model"].get("temperature", 0.0)
    num_samples = config.get("num_samples", 1)
    output_dir = config.get("output_dir", "results/phase1_easy/single")

    dataset_name = config.get("dataset", "humaneval")
    method_name = "single"

    # 2. Task 로드
    print("=" * 60)
    print("📋 Single-Shot Baseline 실험")
    print("=" * 60)
    task = HumanEvalTask()
    print(f"📦 데이터셋: {task}")

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
    samples_to_run = min(num_samples, len(task))

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        print(f"\n--- [{i+1}/{samples_to_run}] {sample.task_id} ---")

        # 프롬프트 구성
        prompt = build_prompt(sample.prompt)

        # 모델 호출 + 시간 측정
        gen_start = time.perf_counter()
        raw_output = model.generate(prompt)
        gen_end = time.perf_counter()

        latency_sec = gen_end - gen_start

        # 코드 추출
        code = extract_code(prompt, raw_output, sample.entry_point)

        # 실행 및 평가
        exec_result = execute_code(code, sample.test, sample.entry_point)
        status = get_status(exec_result)
        error_type, error_message = classify_error(exec_result)

        pretty_status = {
            "pass": "✅ PASS",
            "fail": "❌ FAIL",
            "timeout": "⏱️ TIMEOUT",
        }[status]
        print(f"  {pretty_status}")

        # 결과 기록
        result_entry = {
            "dataset": dataset_name,
            "task_id": sample.task_id,
            "entry_point": sample.entry_point,
            "method": method_name,
            "model_name": model_name,
            "attempt_idx": 0,
            "status": status,
            "passed": exec_result.passed,
            "timeout": exec_result.timeout,
            "raw_output": raw_output,
            "generated_code": code,
            "error_type": error_type,
            "error_message": error_message,
            "latency_sec": latency_sec,
        }
        all_results.append(result_entry)

    # 5. 결과 요약 및 저장
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

    # 결과 저장
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
        },
        os.path.join(output_dir, "summary.json"),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.single <config.yaml>")
        sys.exit(1)
    run_single_shot(sys.argv[1])