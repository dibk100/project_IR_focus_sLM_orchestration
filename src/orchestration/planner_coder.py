"""
Planner-Coder Orchestration

흐름: Problem → D → G → Evaluate
1. task 읽기
2. Planner(Decomposer)가 구현 계획 생성
3. Coder(Generator)가 계획을 바탕으로 코드 생성
4. 실행/평가
5. 결과 저장

Phase 1에서는 repair 없이 planner/coder의 순수 효과를 본다.

핵심:
- Constrained Planner: step 수 최대 5 제한, 간결한 plan 생성
- 2-call 구조 (plan 1회 + code 1회)
- plan step은 stage="plan", is_planner=True로 기록
- code step은 stage="generate"로 기록

Phase1 ver3 기준:
- nested config 구조 사용
- HFModel.generate()의 구조화된 반환값 사용
- step_logs / trajectory_logs / summary / analysis 저장
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

from src.utils.io import save_result, save_results_jsonl,make_run_id, append_jsonl
from src.utils.dataloader import load_task_and_adapter
from src.utils.prompt_loader import build_planner_prompt_for_sample, build_coder_prompt_for_sample
from src.utils.prompting.planner_coder import extract_planner_output



def run_planner_coder(config_path: str):
    """Planner-Coder 실험 실행"""
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

    # planner / coder 모델 설정
    # 동일 모델을 사용하되, planner는 max_new_tokens를 제한할 수 있음
    planner_cfg = model_cfg.get("planner", {})
    coder_cfg = model_cfg.get("coder", {})

    # 공통 모델 이름 (planner/coder가 분리 지정되지 않으면 상위 name 사용)
    base_model_name = model_cfg.get("name", "")
    planner_model_name = planner_cfg.get("name", base_model_name)
    planner_max_new_tokens = planner_cfg.get("max_new_tokens", 256)
    planner_temperature = planner_cfg.get("temperature", model_cfg.get("temperature", 0.0))

    coder_model_name = coder_cfg.get("name", base_model_name)
    coder_max_new_tokens = coder_cfg.get("max_new_tokens", model_cfg.get("max_new_tokens", 512))
    coder_temperature = coder_cfg.get("temperature", model_cfg.get("temperature", 0.0))

    method_name = method_cfg.get("name", "planner_coder")

    max_calls = budget_cfg.get("max_calls", 2)

    output_dir = output_cfg.get("dir", f"results/phase1_ver3/{dataset_name}/planner_coder")

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🧠 Planner-Coder 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"planner_model       : {planner_model_name}")
    print(f"planner_max_tokens  : {planner_max_new_tokens}")
    print(f"coder_model         : {coder_model_name}")
    print(f"coder_max_tokens    : {coder_max_new_tokens}")
    print(f"max_calls           : {max_calls}")
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
    print(f"🔄 Planner 모델 로딩: {planner_model_name}")
    planner_model = HFModel(
        model_name=planner_model_name,
        max_new_tokens=planner_max_new_tokens,
        temperature=planner_temperature,
    )
    print("✅ Planner 모델 로딩 완료")

    # Coder 모델: 동일 모델이면 공유, 아니면 별도 로드
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

    # ── 4. 실험 실행 ──
    # step_logs / trajectory_logs는 메모리 누적 대신 즉시 파일에 기록 (OOM 방지)
    step_log_path = os.path.join(output_dir, "step_logs.jsonl")
    trajectory_log_path = os.path.join(output_dir, "trajectory_logs.jsonl")

    # 이전 실행 파일 초기화 (재실행 시 중복 방지)
    if save_step_level and os.path.exists(step_log_path):
        os.remove(step_log_path)
    if save_trajectory_level and os.path.exists(trajectory_log_path):
        os.remove(trajectory_log_path)

    eval_results = []
    written_steps = 0
    written_trajectories = 0

    # run-level analysis용 집계 (메모리 부담 없는 counter만 유지)
    transition_counts = {}
    failure_type_counts = {}

    # avg 계산용 누적 변수
    sum_tokens = 0.0
    sum_latency = 0.0
    sum_calls = 0

    samples_to_run = min(num_samples, len(task))

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        problem_id = sample.task_id
        trajectory_id = f"{problem_id}_run0"

        print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

        # 문제별 누적 토큰/레이턴시 추적
        cumulative_total_tokens = 0
        cumulative_latency = 0.0
        transition_path = []
        call_count = 0

        # ── 4-1. Planner step (D: Decomposer) ──
        planner_prompt = build_planner_prompt_for_sample(sample)

        planner_start = time.perf_counter()
        planner_gen_result = planner_model.generate(planner_prompt)
        planner_end = time.perf_counter()
        planner_latency = planner_end - planner_start

        planner_raw_text = planner_gen_result["text"]
        planner_input_tokens = planner_gen_result["input_tokens"]
        planner_output_tokens = planner_gen_result["output_tokens"]
        planner_total_tokens = planner_gen_result["total_tokens"]

        call_count += 1
        cumulative_total_tokens += planner_total_tokens
        cumulative_latency += planner_latency

        # planner output 추출
        planner_output = extract_planner_output(planner_raw_text)
        print(f"  📝 Plan: {planner_output[:80]}...")

        # planner step log
        planner_step_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "step_id": 0,
            "call_index": 0,
            "candidate_id": 0,
            "stage": "plan",
            "is_retry": False,
            "is_repair": False,
            "is_planner": True,
            "input_tokens": planner_input_tokens,
            "output_tokens": planner_output_tokens,
            "total_tokens": planner_total_tokens,
            "latency_sec": planner_latency,
            "code": None,
            "planner_output": planner_output if save_code else None,
            "exec_ok": None,
            "test_pass": None,
            "status": "PLAN_DONE",
            "error_type": None,
            "error_stage": None,
            "error_message": None,
            "tests_passed": None,
            "tests_total": None,
            "code_length": 0,
            "selected": None,
            "selection_rank": None,
        }

        if hasattr(sample, "entry_point"):
            planner_step_entry["entry_point"] = sample.entry_point

        if save_step_level:
            append_jsonl(planner_step_entry, step_log_path)
            written_steps += 1

        # ── 4-2. Coder step (G: Generator) ──
        coder_prompt = build_coder_prompt_for_sample(sample, planner_output)

        coder_start = time.perf_counter()
        coder_gen_result = coder_model.generate(
            coder_prompt,
            max_new_tokens=coder_max_new_tokens,
        )
        coder_end = time.perf_counter()
        coder_latency = coder_end - coder_start

        coder_raw_text = coder_gen_result["text"]
        coder_input_tokens = coder_gen_result["input_tokens"]
        coder_output_tokens = coder_gen_result["output_tokens"]
        coder_total_tokens = coder_gen_result["total_tokens"]

        call_count += 1
        cumulative_total_tokens += coder_total_tokens
        cumulative_latency += coder_latency

        # 4-3. 코드 추출 (planner 전용 추출)
        generated_code = adapter.extract_code_for_planner(
            sample,
            coder_raw_text,
        )

        # 4-4. 실행 / 평가
        exec_result = adapter.execute(sample, generated_code)
        eval_results.append(exec_result)

        # 4-5. attempt record 생성
        attempt_record = adapter.make_attempt_record(
            sample=sample,
            method=method_name,
            model_name=f"{planner_model_name} -> {coder_model_name}",
            attempt_idx=0,
            prompt=coder_prompt,
            raw_output=coder_raw_text,
            generated_code=generated_code,
            latency_sec=cumulative_latency,
            exec_result=exec_result,
        )

        current_status = attempt_record.status
        transition_path.append("PLAN_DONE")
        transition_path.append(current_status)

        num_exec_fail = 1 if str(current_status).startswith("EXEC_FAIL") else 0
        num_test_fail = 1 if str(current_status).startswith("TEST_FAIL") else 0

        # coder step log
        coder_step_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "step_id": 1,
            "call_index": 1,
            "candidate_id": 0,
            "stage": "generate",
            "is_retry": False,
            "is_repair": False,
            "is_planner": False,
            "input_tokens": coder_input_tokens,
            "output_tokens": coder_output_tokens,
            "total_tokens": coder_total_tokens,
            "latency_sec": coder_latency,
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
            coder_step_entry["entry_point"] = sample.entry_point

        if save_step_level:
            append_jsonl(coder_step_entry, step_log_path)
            written_steps += 1

        pretty_status = (
            "✅ PASS"
            if current_status == "PASS"
            else f"❌ {current_status}"
        )
        print(f"  {pretty_status}")

        # 4-6. trajectory-level log
        trajectory_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "num_steps": call_count,
            "call_count": call_count,
            "final_status": current_status,
            "final_tests_passed": attempt_record.tests_passed,
            "final_tests_total": attempt_record.tests_total,
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

        # avg 계산용 누적
        sum_tokens += trajectory_entry["total_tokens"]
        sum_latency += trajectory_entry["total_latency"]
        sum_calls += trajectory_entry["call_count"]

        # run-level analysis 집계 (메모리 내 counter만 유지)
        path = trajectory_entry["transition_path"]
        for j in range(len(path) - 1):
            coarse_a = path[j].split(":")[0]
            coarse_b = path[j + 1].split(":")[0]
            key = f"{coarse_a}->{coarse_b}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

        status = coder_step_entry["status"]
        if status not in ("PASS", "PLAN_DONE"):
            failure_type_counts[status] = failure_type_counts.get(status, 0) + 1

        # CUDA 캐시 비우기 + CPU RAM GC (OOM 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del planner_step_entry, coder_step_entry, trajectory_entry
        del attempt_record, exec_result, generated_code
        del planner_output, planner_raw_text, coder_raw_text
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

    # ── 7. Run-level analysis summary ──
    # transition_counts / failure_type_counts는 루프 중 이미 집계됨
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
        print("Usage: python -m src.orchestration.planner_coder <config.yaml>")
        sys.exit(1)

    run_planner_coder(sys.argv[1])