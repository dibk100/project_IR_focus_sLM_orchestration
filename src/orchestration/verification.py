"""
Verification-Only Orchestration

흐름:
1. task 읽기
2. initial generation
3. 실행/평가 (ground truth)
4. verifier가 코드가 맞는지 PASS/FAIL 판정
5. 결과 저장

핵심:
- verifier는 수정하지 않고 판정만 한다.
- 즉 pure verification (V) 효과를 측정한다.
"""
import os
import re
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


def build_verification_prompt(task_prompt: str, candidate_code: str) -> str:
    """코드의 정답 여부만 판정하도록 유도."""
    return f"""You are verifying whether a Python solution is correct for a programming task.

Task:
{task_prompt}

Candidate Solution:
{candidate_code}

Instructions:
- Decide whether the candidate solution is likely correct.
- Answer with exactly one label on the first line: PASS or FAIL
- On the second line, give one short reason.
- Do not write code.
- Do not rewrite the solution.

Verdict:
"""


def parse_verdict(verifier_output: str) -> str:
    """
    verifier 출력에서 PASS / FAIL 추출
    기본값은 FAIL로 둔다.
    """
    text = verifier_output.strip()
    first_line = text.splitlines()[0].strip().upper() if text else ""

    if "PASS" in first_line:
        return "PASS"
    if "FAIL" in first_line:
        return "FAIL"

    # fallback: 전체 텍스트 검색
    upper = text.upper()
    pass_pos = upper.find("PASS")
    fail_pos = upper.find("FAIL")

    if pass_pos != -1 and fail_pos != -1:
        return "PASS" if pass_pos < fail_pos else "FAIL"
    if pass_pos != -1:
        return "PASS"
    return "FAIL"


def verifier_matches_ground_truth(verdict: str, exec_result) -> bool:
    gt = "PASS" if exec_result.passed else "FAIL"
    return verdict == gt


def run_verification(config_path: str):
    """verification-only baseline 실행."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    generator_model_name = config["generator_model"]["name"]
    generator_max_new_tokens = config["generator_model"].get("max_new_tokens", 512)
    generator_temperature = config["generator_model"].get("temperature", 0.0)

    verifier_model_name = config["verifier_model"]["name"]
    verifier_max_new_tokens = config["verifier_model"].get("max_new_tokens", 128)
    verifier_temperature = config["verifier_model"].get("temperature", 0.0)

    num_samples = config.get("num_samples", 1)
    output_dir = config.get("output_dir", "results/phase1_easy/verification")
    dataset_name = config.get("dataset", "humaneval")
    method_name = "verification"

    print("=" * 60)
    print("🔍 Verification-Only 실험")
    print("=" * 60)

    task = HumanEvalTask()
    print(f"📦 데이터셋: {task}")

    print(f"🔄 Generator 모델 로딩: {generator_model_name}")
    generator_model = HFModel(
        model_name=generator_model_name,
        max_new_tokens=generator_max_new_tokens,
        temperature=generator_temperature,
    )
    print("✅ Generator 모델 로딩 완료")

    if verifier_model_name == generator_model_name:
        verifier_model = generator_model
        print(f"🔁 Verifier 모델은 Generator와 동일 모델 공유: {verifier_model_name}")
    else:
        print(f"🔄 Verifier 모델 로딩: {verifier_model_name}")
        verifier_model = HFModel(
            model_name=verifier_model_name,
            max_new_tokens=verifier_max_new_tokens,
            temperature=verifier_temperature,
        )
        print("✅ Verifier 모델 로딩 완료")

    samples_to_run = min(num_samples, len(task))
    all_results = []

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        print(f"\n--- [{i+1}/{samples_to_run}] {sample.task_id} ---")

        # 1) initial generation
        gen_prompt = build_prompt(sample.prompt)
        gen_start = time.perf_counter()
        raw_output = generator_model.generate(gen_prompt)
        gen_end = time.perf_counter()
        gen_latency_sec = gen_end - gen_start

        code = extract_code(sample.prompt, raw_output, sample.entry_point)

        # 2) ground truth execution
        exec_result = execute_code(code, sample.test, sample.entry_point)
        status = get_status(exec_result)
        error_type, error_message = classify_error(exec_result)

        # 3) verification
        ver_prompt = build_verification_prompt(sample.prompt, code)
        ver_start = time.perf_counter()
        verifier_output = verifier_model.generate(ver_prompt)
        ver_end = time.perf_counter()
        ver_latency_sec = ver_end - ver_start

        verifier_verdict = parse_verdict(verifier_output)
        verifier_correct = verifier_matches_ground_truth(verifier_verdict, exec_result)

        pretty_status = {
            "pass": "✅ PASS",
            "fail": "❌ FAIL",
            "timeout": "⏱️ TIMEOUT",
        }[status]

        pretty_verifier = "✅ CORRECT" if verifier_correct else "❌ WRONG"
        print(f"  generation: {pretty_status} | verifier: {verifier_verdict} ({pretty_verifier})")

        result_entry = {
            "dataset": dataset_name,
            "task_id": sample.task_id,
            "entry_point": sample.entry_point,
            "method": method_name,
            "generator_model_name": generator_model_name,
            "verifier_model_name": verifier_model_name,
            "status": status,
            "passed": exec_result.passed,
            "timeout": exec_result.timeout,
            "raw_output": raw_output,
            "generated_code": code,
            "error_type": error_type,
            "error_message": error_message,
            "generator_prompt": gen_prompt,
            "verifier_prompt": ver_prompt,
            "verifier_output": verifier_output,
            "verifier_verdict": verifier_verdict,
            "verifier_correct": verifier_correct,
            "generator_latency_sec": gen_latency_sec,
            "verifier_latency_sec": ver_latency_sec,
            "latency_sec": gen_latency_sec + ver_latency_sec,
        }
        all_results.append(result_entry)

    # generation summary
    exec_results = [
        type("R", (), {"passed": r["passed"], "timeout": r["timeout"]})()
        for r in all_results
    ]
    summary = summarize_results(exec_results)

    verifier_accuracy = (
        sum(1 for r in all_results if r["verifier_correct"]) / len(all_results)
        if all_results else 0.0
    )

    print(f"\n{'=' * 60}")
    print("📊 결과 요약")
    print(f"  총 문제: {summary['total']}")
    print(f"  generation 통과: {summary['passed']}")
    print(f"  generation 실패: {summary['failed']}")
    print(f"  generation 타임아웃: {summary['timed_out']}")
    print(f"  generation pass@1: {summary['pass@1']:.4f}")
    print(f"  verifier accuracy: {verifier_accuracy:.4f}")
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
                "orchestration": "verification-only",
                "task": dataset_name,
                "timestamp": timestamp,
                "config_path": config_path,
            },
            "generator_model": {
                "name": generator_model_name,
                "max_new_tokens": generator_max_new_tokens,
                "temperature": generator_temperature,
            },
            "verifier_model": {
                "name": verifier_model_name,
                "max_new_tokens": verifier_max_new_tokens,
                "temperature": verifier_temperature,
            },
            "summary": summary,
            "verification": {
                "verifier_accuracy": verifier_accuracy,
            },
        },
        os.path.join(output_dir, "summary.json"),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.verification <config.yaml>")
        sys.exit(1)

    run_verification(sys.argv[1])