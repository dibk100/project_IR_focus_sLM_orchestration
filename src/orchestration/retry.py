"""
Retry (Refinement-only) Orchestration

흐름:
1. task 읽기
2. initial generation
3. 실행/평가 → 실패 시, 이전 코드를 기반으로 refinement (error message 없이)
4. 반복 (최대 max_retry 횟수까지)

repair와의 차이:
- retry: error message 없이 pure refinement (이전 코드 + 원래 문제만 전달)
- repair: error message를 포함한 feedback 기반 수정

핵심:
- execution feedback(에러 메시지)를 사용하지 않음
- pure refinement 효과 측정

Phase1 ver3 기준:
- nested config 구조 사용
- HFModel.generate()의 구조화된 반환값 사용
- step_logs / trajectory_logs / summary / analysis 저장
"""
import gc
import os
import time
import yaml
from datetime import datetime

from src.models.hf_model import HFModel

from src.evaluation.metrics import (
    summarize_phase1_results,
    summarize_mbpp_failure_breakdown,
)

import torch
from src.utils.io import save_result, save_results_jsonl, append_jsonl, make_run_id
from src.utils.dataloader import load_task_and_adapter

def run_retry(config_path: str):
    """Retry (refinement-only) 실험 실행"""
    # ── 1. Config 로드 ──
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

    method_name = method_cfg.get("name", "retry_only")
    use_previous_code = method_cfg.get("use_previous_code", True)

    max_calls = budget_cfg.get("max_calls", 3)
    max_retry = budget_cfg.get("max_retry", 2)

    output_dir = output_cfg.get("dir", f"results/phase1_ver3/{dataset_name}/retry")

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🔁 Retry (Refinement-only) 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"max_calls           : {max_calls}")
    print(f"max_retry           : {max_retry}")
    print(f"use_previous_code   : {use_previous_code}")
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

    # ── 2. Task / Adapter 로드 ──
    task, adapter = load_task_and_adapter(dataset_name)
    print(f"📦 데이터셋: {dataset_name} | size={len(task)}")

    # ── 3. 모델 로드 ──
    print(f"🔄 모델 로딩: {model_name}")
    model = HFModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    print("✅ 모델 로딩 완료")

    # ── 4. 실험 실행 ──
    # step_logs / trajectory_logs는 메모리 누적 대신 즉시 파일에 기록 (OOM 방지)
    step_log_path = os.path.join(output_dir, "step_logs.jsonl")
    trajectory_log_path = os.path.join(output_dir, "trajectory_logs.jsonl")

    if save_step_level and os.path.exists(step_log_path):
        os.remove(step_log_path)
    if save_trajectory_level and os.path.exists(trajectory_log_path):
        os.remove(trajectory_log_path)

    eval_results = []
    written_steps = 0
    written_trajectories = 0
    transition_counts = {}
    failure_type_counts = {}
    sum_tokens = 0.0
    sum_latency = 0.0
    sum_calls = 0

    samples_to_run = min(num_samples, len(task))

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        problem_id = sample.task_id
        trajectory_id = f"{problem_id}_run0"

        print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

        # 문제별 누적 토큰/레이턴시/상태 추적
        cumulative_input_tokens = 0
        cumulative_output_tokens = 0
        cumulative_total_tokens = 0
        cumulative_latency = 0.0
        transition_path = []
        num_exec_fail = 0
        num_test_fail = 0
        call_count = 0

        previous_code = None
        final_attempt_record = None

        # initial prompt
        current_prompt = adapter.build_initial_prompt(sample)

        for attempt_idx in range(max_retry + 1):
            is_retry = attempt_idx > 0

            # 4-1. 모델 호출 + 시간 측정
            gen_start = time.perf_counter()
            gen_result = model.generate(current_prompt)
            gen_end = time.perf_counter()
            latency_sec = gen_end - gen_start

            raw_text = gen_result["text"]
            input_tokens = gen_result["input_tokens"]
            output_tokens = gen_result["output_tokens"]
            total_tokens = gen_result["total_tokens"]

            call_count += 1
            cumulative_input_tokens += input_tokens
            cumulative_output_tokens += output_tokens
            cumulative_total_tokens += total_tokens
            cumulative_latency += latency_sec

            # 4-2. 코드 추출
            generated_code = adapter.extract_code(sample, raw_text)

            # 4-3. 실행 / 평가
            exec_result = adapter.execute(sample, generated_code)

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

            final_attempt_record = attempt_record
            current_status = attempt_record.status
            transition_path.append(current_status)

            if str(current_status).startswith("EXEC_FAIL"):
                num_exec_fail += 1
            if str(current_status).startswith("TEST_FAIL"):
                num_test_fail += 1

            # 4-5. step-level log
            step_entry = {
                "run_id": run_id,
                "dataset": dataset_name,
                "problem_id": problem_id,
                "method": method_name,
                "trajectory_id": trajectory_id,
                "step_id": attempt_idx,
                "call_index": attempt_idx,
                "candidate_id": 0,
                "stage": "generate" if not is_retry else "retry",
                "is_retry": is_retry,
                "is_repair": False,
                "is_planner": False,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_sec": latency_sec,
                "code": generated_code if save_code else None,
                "exec_ok": attempt_record.exec_ok,
                "test_pass": attempt_record.test_pass,
                "status": current_status,
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

            if save_step_level:
                append_jsonl(step_entry, step_log_path)
                written_steps += 1

            pretty_status = (
                "✅ PASS"
                if current_status == "PASS"
                else f"❌ {current_status}"
            )
            print(f"  attempt {attempt_idx}: {pretty_status}")

            # 성공하면 종료
            if attempt_record.passed:
                break

            # 마지막 attempt이면 종료
            if attempt_idx == max_retry:
                break

            # 4-6. 다음 retry를 위한 refinement prompt 구성
            previous_code = generated_code
            current_prompt = adapter.build_refinement_prompt(
                sample=sample,
                previous_code=previous_code,
            )


        # (문제 종료) 최종 exec_result를 eval_results에 추가
        eval_results.append(exec_result)

        # 4-7. trajectory-level log
        trajectory_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "num_steps": call_count,
            "call_count": call_count,
            "final_status": final_attempt_record.status,
            "final_tests_passed": final_attempt_record.tests_passed,
            "final_tests_total": final_attempt_record.tests_total,
            "total_tokens": cumulative_total_tokens,
            "total_latency": cumulative_latency,
            "num_exec_fail": num_exec_fail,
            "num_test_fail": num_test_fail,
            "transition_path": transition_path,
            "budget_used": {
                "tokens": cumulative_total_tokens,
                "calls": call_count,
                "latency": cumulative_latency,
            },
        }

        if save_trajectory_level:
            append_jsonl(trajectory_entry, trajectory_log_path)
            written_trajectories += 1

        # avg 누적
        sum_tokens += trajectory_entry["total_tokens"]
        sum_latency += trajectory_entry["total_latency"]
        sum_calls += trajectory_entry["call_count"]

        # run-level analysis 집계
        path = trajectory_entry["transition_path"]
        for j in range(len(path) - 1):
            coarse_a = path[j].split(":")[0]
            coarse_b = path[j + 1].split(":")[0]
            key = f"{coarse_a}->{coarse_b}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

        status = step_entry["status"]
        if status != "PASS":
            failure_type_counts[status] = failure_type_counts.get(status, 0) + 1

        # CUDA 캐시 비우기 (OOM 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── 5. 결과 요약 ──
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

    # ── 6. Problem-level summary ──
    n = len(eval_results)
    avg_tokens = sum_tokens / n if n else 0.0
    avg_latency = sum_latency / n if n else 0.0
    avg_calls = sum_calls / n if n else 0.0

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

    # ── 7. Run-level analysis summary (루프 중 이미 집계됨) ──
    run_analysis = {
        "run_id": run_id,
        "dataset": dataset_name,
        "method": method_name,
        "transition_counts": transition_counts,
        "failure_type_counts": failure_type_counts,
    }

    # ── 8. 결과 저장 ──
    # step_logs / trajectory_logs는 루프 중 streaming으로 이미 기록됨
    if save_step_level:
        print(f"💾 step_logs 기록 완료: {step_log_path} ({written_steps}건)")

    if save_trajectory_level:
        print(f"💾 trajectory_logs 기록 완료: {trajectory_log_path} ({written_trajectories}건)")

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
        print("Usage: python -m src.orchestration.retry <config.yaml>")
        sys.exit(1)

    run_retry(sys.argv[1])