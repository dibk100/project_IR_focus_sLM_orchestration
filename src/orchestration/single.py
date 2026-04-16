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
"""
Single-Shot Orchestration
task 읽기 -> 모델 호출 -> 코드 추출 -> 실행/평가 -> 결과 저장

Phase1 ver3 기준:
- nested config 구조 사용
- HFModel.generate()의 구조화된 반환값 사용
- step_logs / trajectory_logs / summary 저장
"""
import gc
import os
import time
import yaml

import torch

from src.models.hf_model import HFModel

from src.evaluation.metrics import (
    summarize_phase1_results,
    summarize_mbpp_failure_breakdown,
)

from src.utils.io import save_result, save_results_jsonl,make_run_id
from src.utils.dataloader import load_task_and_adapter


def run_single_shot(config_path: str):
    """Single-shot baseline 실험 실행"""
    # 1. Config 로드
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg = config.get("run", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    method_cfg = config.get("method", {})
    budget_cfg = config.get("budget", {})
    output_cfg = config.get("output", {})
    logging_cfg = config.get("logging", {})

    run_id = make_run_id(config)
    seed = run_cfg.get("seed", 42)

    dataset_name = dataset_cfg.get("name", "humaneval")
    num_samples = dataset_cfg.get("num_samples", 1)

    model_name = model_cfg["name"]
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    temperature = model_cfg.get("temperature", 0.0)

    method_name = method_cfg.get("name", "single_shot")
    max_calls = budget_cfg.get("max_calls", 1)

    output_dir = output_cfg.get("dir", f"results/phase1_ver3/{dataset_name}/single")

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("📋 Single-Shot Baseline 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"max_new_tokens      : {max_new_tokens}")
    print(f"temperature         : {temperature}")
    print(f"seed                : {seed}")
    print(f"output_dir          : {output_dir}")
    print("=" * 60)

    # config snapshot 저장
    save_result(
        {
            "run": {
                "run_id": run_id,
                "seed": seed,
            },
            "dataset": dataset_cfg,
            "model": model_cfg,
            "method": method_cfg,
            "budget": budget_cfg,
            "output": output_cfg,
            "logging": logging_cfg,
            "config_path": config_path,
        },
        os.path.join(output_dir, "config.json"),
    )

    # 2. Task / Adapter 로드
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
    step_logs = []
    trajectory_logs = []
    eval_results = []

    samples_to_run = min(num_samples, len(task))

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        problem_id = sample.task_id
        trajectory_id = f"{problem_id}_run0"

        print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

        # 4-1. prompt 구성
        prompt = adapter.build_initial_prompt(sample)

        # 4-2. 모델 호출 + 시간 측정
        gen_start = time.perf_counter()
        gen_result = model.generate(prompt)
        gen_end = time.perf_counter()
        latency_sec = gen_end - gen_start

        raw_text = gen_result["text"]
        input_tokens = gen_result["input_tokens"]
        output_tokens = gen_result["output_tokens"]
        total_tokens = gen_result["total_tokens"]

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

        # 4-6. step-level log
        step_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "step_id": 0,
            "call_index": 0,
            "candidate_id": 0,
            "stage": "generate",
            "is_retry": False,
            "is_repair": False,
            "is_planner": False,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_sec": latency_sec,
            "code": generated_code if save_code else None,
            "exec_ok": attempt_record.exec_ok,
            "test_pass": attempt_record.test_pass,
            "status": attempt_record.status,
            "error_type": attempt_record.error_type,
            "error_stage": attempt_record.error_stage,
            "error_message": attempt_record.error_message,
            "tests_passed": attempt_record.tests_passed,
            "tests_total": attempt_record.tests_total,
            "code_length": len(generated_code) if generated_code is not None else 0,
            "selected": None,
            "selection_rank": None,
        }

        if hasattr(sample, "entry_point"):
            step_entry["entry_point"] = sample.entry_point

        step_logs.append(step_entry)

        # 4-7. trajectory-level log
        final_status = attempt_record.status
        num_exec_fail = 1 if str(final_status).startswith("EXEC_FAIL") else 0
        num_test_fail = 1 if str(final_status).startswith("TEST_FAIL") else 0

        trajectory_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "num_steps": 1,
            "call_count": 1,
            "final_status": final_status,
            "final_tests_passed": attempt_record.tests_passed,
            "final_tests_total": attempt_record.tests_total,
            "total_tokens": total_tokens,
            "total_latency": latency_sec,
            "num_exec_fail": num_exec_fail,
            "num_test_fail": num_test_fail,
            "transition_path": [final_status],
            "budget_used": {
                "tokens": total_tokens,
                "calls": 1,
                "latency": latency_sec,
            },
        }

        trajectory_logs.append(trajectory_entry)

        pretty_status = (
            "✅ PASS"
            if final_status == "PASS"
            else f"❌ {final_status}"
        )
        print(f"  {pretty_status}")
        
        # OOM 방지
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del gen_result, step_entry, trajectory_entry
        del attempt_record, exec_result
        gc.collect()

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
    extra_summary = summarize_mbpp_failure_breakdown(eval_results)
    print("📌 Failure Breakdown")
    print(f"  code_failed: {extra_summary['code_failed']}")
    print(f"  setup_failed: {extra_summary['setup_failed']}")
    print(f"  test_failed: {extra_summary['test_failed']}")
    print(f"  semantic_failed: {extra_summary['semantic_failed']}")
    print(f"  execution_failed: {extra_summary['execution_failed']}")
    print(f"{'=' * 60}")

    # 6. Problem-level summary
    avg_tokens = (
        sum(x["total_tokens"] for x in trajectory_logs) / len(trajectory_logs)
        if trajectory_logs else 0.0
    )
    avg_latency = (
        sum(x["total_latency"] for x in trajectory_logs) / len(trajectory_logs)
        if trajectory_logs else 0.0
    )
    avg_calls = (
        sum(x["call_count"] for x in trajectory_logs) / len(trajectory_logs)
        if trajectory_logs else 0.0
    )

    problem_summary = {
        "run_id": run_id,
        "dataset": dataset_name,
        "method": method_name,
        "total_problems": summary["total"],
        "num_pass": summary["passed"],
        "pass_at_1": summary["pass@1"],
        "execution_success_rate": summary["execution_success_rate"],
        "conditional_pass": summary["conditional_pass"],
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "avg_calls": avg_calls,
        "extra_summary": extra_summary,
    }

    # 7. Run-level analysis summary
    transition_counts = {}
    failure_type_counts = {}

    for traj in trajectory_logs:
        path = traj["transition_path"]
        for j in range(len(path) - 1):
            coarse_a = path[j].split(":")[0]
            coarse_b = path[j + 1].split(":")[0]
            key = f"{coarse_a}->{coarse_b}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

    for step in step_logs:
        status = step["status"]
        if status != "PASS":
            failure_type_counts[status] = failure_type_counts.get(status, 0) + 1

    run_analysis = {
        "run_id": run_id,
        "dataset": dataset_name,
        "method": method_name,
        "transition_counts": transition_counts,
        "failure_type_counts": failure_type_counts,
    }

    # 8. 결과 저장
    if save_step_level:
        save_results_jsonl(
            step_logs,
            os.path.join(output_dir, "step_logs.jsonl"),
        )

    if save_trajectory_level:
        save_results_jsonl(
            trajectory_logs,
            os.path.join(output_dir, "trajectory_logs.jsonl"),
        )

    if save_problem_summary:
        save_result(
            problem_summary,
            os.path.join(output_dir, "summary.json"),
        )

    if save_run_analysis:
        save_result(
            run_analysis,
            os.path.join(output_dir, "analysis.json"),
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.single <config.yaml>")
        sys.exit(1)

    run_single_shot(sys.argv[1])