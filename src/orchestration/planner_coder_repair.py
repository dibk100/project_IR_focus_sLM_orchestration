"""
Planner-Coder-Repair Orchestration

흐름: Problem → D(Plan) → G(Generate) → V(Evaluate) → [R(Repair) → V] × N
1. task 읽기
2. Planner(Decomposer)가 구현 계획 생성          [stage="plan"]
3. Coder(Generator)가 계획 기반 초기 코드 생성    [stage="generate"]
4. 실행/평가 (Verification)
5. 실패 시: (문제 + plan + 이전코드 + 에러)로 Repair prompt → 재생성  [stage="repair"]
6. 최대 max_repair 횟수까지 반복
7. 결과 저장

planner_coder와의 차이:
- planner_coder: plan 1회 + code 1회 (2 calls, repair 없음)
- planner_coder_repair: plan 1회 + code 1회 + repair N회 (2+N calls)

repair와의 차이:
- repair: plan 없이 error feedback만 활용
- planner_coder_repair: plan context를 repair에도 유지하여 구조 수정 가이드 제공

핵심:
- Planner: max_new_tokens=256 (간결한 plan)
- Coder (initial): max_new_tokens=512
- Coder (repair): plan + previous_code + error_message 포함
- step_logs / trajectory_logs / summary / analysis 저장

Phase1 ver3 기준:
- nested config 구조 사용
- HFModel.generate()의 구조화된 반환값 사용
- gc.collect() + cuda.empty_cache()로 OOM 방지
"""
import gc
import os
import time
import yaml
from datetime import datetime

import torch

from src.models.hf_model import HFModel

from src.tasks.humaneval import HumanEvalTask, HumanEvalSample
from src.tasks.mbpp import MBPPTask, MBPPSample

from src.adapters.humaneval import HumanEvalAdapter
from src.adapters.mbpp import MBPPAdapter

from src.evaluation.metrics import (
    summarize_phase1_results,
    summarize_mbpp_failure_breakdown,
)

from src.utils.io import save_result, save_results_jsonl, append_jsonl
from src.utils.prompting.planner_coder import (
    build_humaneval_planner_prompt,
    build_mbpp_planner_prompt,
    build_humaneval_coder_prompt,
    build_mbpp_coder_prompt,
    extract_planner_output,
)
from src.utils.prompting.planner_coder_repair import (
    build_humaneval_repair_with_plan_prompt,
    build_mbpp_repair_with_plan_prompt,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_task_and_adapter(dataset_name: str):
    if dataset_name == "humaneval":
        return HumanEvalTask(), HumanEvalAdapter()
    if dataset_name == "mbpp":
        return MBPPTask(), MBPPAdapter()
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_run_id(config: dict) -> str:
    base_run_id = config.get("run", {}).get("run_id", "phase1_planner_coder_repair")
    suffix = datetime.now().strftime("%m%d%H%M%S")
    return f"{base_run_id}_{suffix}"


def build_planner_prompt_for_sample(sample) -> str:
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_planner_prompt(sample)
    if isinstance(sample, MBPPSample):
        return build_mbpp_planner_prompt(sample)
    raise TypeError(f"Unsupported sample type: {type(sample)}")


def build_coder_prompt_for_sample(sample, planner_output: str) -> str:
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_coder_prompt(sample, planner_output)
    if isinstance(sample, MBPPSample):
        return build_mbpp_coder_prompt(sample, planner_output)
    raise TypeError(f"Unsupported sample type: {type(sample)}")


def build_repair_prompt_for_sample(
    sample, planner_output: str, previous_code: str, error_message: str
) -> str:
    if isinstance(sample, HumanEvalSample):
        return build_humaneval_repair_with_plan_prompt(
            sample, planner_output, previous_code, error_message
        )
    if isinstance(sample, MBPPSample):
        return build_mbpp_repair_with_plan_prompt(
            sample, planner_output, previous_code, error_message
        )
    raise TypeError(f"Unsupported sample type: {type(sample)}")


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def run_planner_coder_repair(config_path: str):
    """Planner-Coder-Repair 실험 실행"""

    # ── 1. Config 로드 ──
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg     = config.get("run", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg   = config.get("model", {})
    method_cfg  = config.get("method", {})
    budget_cfg  = config.get("budget", {})
    output_cfg  = config.get("output", {})
    logging_cfg = config.get("logging", {})

    run_id       = make_run_id(config)
    seed         = run_cfg.get("seed", 42)
    dataset_name = dataset_cfg.get("name", "humaneval")
    num_samples  = dataset_cfg.get("num_samples", 1)

    # nested model config (planner / coder / repairer)
    planner_cfg = model_cfg.get("planner", {})
    coder_cfg   = model_cfg.get("coder", {})
    # repairer: coder와 같은 모델을 기본으로 사용 (별도 지정 가능)
    repairer_cfg = model_cfg.get("repairer", coder_cfg)

    planner_name       = planner_cfg.get("name", model_cfg.get("name"))
    planner_tokens     = planner_cfg.get("max_new_tokens", 256)
    planner_temp       = planner_cfg.get("temperature", model_cfg.get("temperature", 0.2))

    coder_name         = coder_cfg.get("name", model_cfg.get("name"))
    coder_tokens       = coder_cfg.get("max_new_tokens", 512)
    coder_temp         = coder_cfg.get("temperature", model_cfg.get("temperature", 0.2))

    repairer_name      = repairer_cfg.get("name", coder_name)
    repairer_tokens    = repairer_cfg.get("max_new_tokens", coder_tokens)
    repairer_temp      = repairer_cfg.get("temperature", coder_temp)

    method_name  = method_cfg.get("name", "planner_coder_repair")
    max_repair   = budget_cfg.get("max_repair", 2)
    # max_calls는 참고용 (실제: 2 + max_repair)
    max_calls    = budget_cfg.get("max_calls", 2 + max_repair)

    output_dir   = output_cfg.get("dir", f"results/phase1_ver3/{dataset_name}/planner_coder_repair")

    save_step_level       = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary  = logging_cfg.get("save_problem_summary", True)
    save_run_analysis     = logging_cfg.get("save_run_analysis", True)
    save_code             = logging_cfg.get("save_code", True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🔧 Planner-Coder-Repair 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"max_repair          : {max_repair}  (total calls: 2+{max_repair}={2+max_repair})")
    print(f"planner             : {planner_name} / tokens={planner_tokens}")
    print(f"coder               : {coder_name} / tokens={coder_tokens}")
    print(f"repairer            : {repairer_name} / tokens={repairer_tokens}")
    print(f"seed                : {seed}")
    print(f"output_dir          : {output_dir}")
    print("=" * 60)

    # config snapshot 저장
    save_result(
        {
            "run": {"run_id": run_id, "seed": seed},
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

    # ── 2. Task / Adapter ──
    task, adapter = load_task_and_adapter(dataset_name)
    print(f"📦 데이터셋: {dataset_name} | size={len(task)}")

    # ── 3. 모델 로드 ──
    # 같은 모델명이면 인스턴스를 공유해 VRAM 절약
    print(f"🔄 Planner 모델 로딩: {planner_name}")
    planner_model = HFModel(
        model_name=planner_name,
        max_new_tokens=planner_tokens,
        temperature=planner_temp,
    )

    if coder_name == planner_name:
        print("✅ Coder/Repairer = Planner (같은 모델, 인스턴스 공유)")
        coder_model    = planner_model
        repairer_model = planner_model
    else:
        print(f"🔄 Coder 모델 로딩: {coder_name}")
        coder_model = HFModel(
            model_name=coder_name,
            max_new_tokens=coder_tokens,
            temperature=coder_temp,
        )
        if repairer_name == coder_name:
            repairer_model = coder_model
        else:
            print(f"🔄 Repairer 모델 로딩: {repairer_name}")
            repairer_model = HFModel(
                model_name=repairer_name,
                max_new_tokens=repairer_tokens,
                temperature=repairer_temp,
            )

    # max_new_tokens를 동적으로 오버라이드하기 위한 헬퍼
    def _gen(model: HFModel, prompt: str, max_tokens: int) -> dict:
        orig = model.max_new_tokens
        model.max_new_tokens = max_tokens
        result = model.generate(prompt)
        model.max_new_tokens = orig
        return result

    print("✅ 모델 로딩 완료")

    # ── 4. 실험 루프 ──
    step_log_path       = os.path.join(output_dir, "step_logs.jsonl")
    trajectory_log_path = os.path.join(output_dir, "trajectory_logs.jsonl")

    if save_step_level and os.path.exists(step_log_path):
        os.remove(step_log_path)
    if save_trajectory_level and os.path.exists(trajectory_log_path):
        os.remove(trajectory_log_path)

    eval_results      = []
    written_steps     = 0
    written_trajectories = 0
    transition_counts = {}
    failure_type_counts = {}
    sum_tokens   = 0.0
    sum_latency  = 0.0
    sum_calls    = 0

    samples_to_run = min(num_samples, len(task))
    global_step_id = 0  # 전역 step counter

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        problem_id   = sample.task_id
        trajectory_id = f"{problem_id}_run0"

        print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

        cumulative_total_tokens = 0
        cumulative_latency      = 0.0
        transition_path         = []
        num_exec_fail           = 0
        num_test_fail           = 0
        call_count              = 0

        # ── 4-1. Planner step ──────────────────────────
        planner_prompt = build_planner_prompt_for_sample(sample)

        t0 = time.perf_counter()
        planner_result = _gen(planner_model, planner_prompt, planner_tokens)
        planner_latency = time.perf_counter() - t0

        planner_raw_text = planner_result["text"]
        planner_output   = extract_planner_output(planner_raw_text)

        call_count += 1
        cumulative_total_tokens += planner_result["total_tokens"]
        cumulative_latency      += planner_latency

        planner_step_entry = {
            "run_id":          run_id,
            "dataset":         dataset_name,
            "problem_id":      problem_id,
            "method":          method_name,
            "trajectory_id":   trajectory_id,
            "step_id":         global_step_id,
            "call_index":      0,
            "candidate_id":    0,
            "stage":           "plan",
            "is_retry":        False,
            "is_repair":       False,
            "is_planner":      True,
            "input_tokens":    planner_result["input_tokens"],
            "output_tokens":   planner_result["output_tokens"],
            "total_tokens":    planner_result["total_tokens"],
            "latency_sec":     planner_latency,
            "code":            None,
            "exec_ok":         None,
            "test_pass":       None,
            "status":          "PLAN_DONE",
            "error_type":      None,
            "error_stage":     None,
            "error_message":   None,
            "tests_passed":    None,
            "tests_total":     None,
            "code_length":     0,
            "selected":        None,
            "selection_rank":  None,
            "plan_text":       planner_output,
        }
        if hasattr(sample, "entry_point"):
            planner_step_entry["entry_point"] = sample.entry_point

        if save_step_level:
            append_jsonl(planner_step_entry, step_log_path)
            written_steps += 1

        global_step_id += 1
        print(f"  📝 Plan: {planner_output[:80].replace(chr(10), ' ')}...")

        # ── 4-2. Initial Code generation ───────────────
        coder_prompt = build_coder_prompt_for_sample(sample, planner_output)

        t0 = time.perf_counter()
        coder_result  = _gen(coder_model, coder_prompt, coder_tokens)
        coder_latency = time.perf_counter() - t0

        coder_raw_text   = coder_result["text"]
        generated_code   = adapter.extract_code(sample, coder_raw_text)
        exec_result      = adapter.execute(sample, generated_code)
        attempt_record   = adapter.make_attempt_record(
            sample=sample,
            method=method_name,
            model_name=coder_name,
            attempt_idx=0,
            prompt=coder_prompt,
            raw_output=coder_raw_text,
            generated_code=generated_code,
            latency_sec=coder_latency,
            exec_result=exec_result,
        )

        call_count += 1
        cumulative_total_tokens += coder_result["total_tokens"]
        cumulative_latency      += coder_latency

        current_status = attempt_record.status
        transition_path.append(current_status)
        if str(current_status).startswith("EXEC_FAIL"):
            num_exec_fail += 1
        if str(current_status).startswith("TEST_FAIL"):
            num_test_fail += 1

        coder_step_entry = {
            "run_id":          run_id,
            "dataset":         dataset_name,
            "problem_id":      problem_id,
            "method":          method_name,
            "trajectory_id":   trajectory_id,
            "step_id":         global_step_id,
            "call_index":      1,
            "candidate_id":    0,
            "stage":           "generate",
            "is_retry":        False,
            "is_repair":       False,
            "is_planner":      False,
            "input_tokens":    coder_result["input_tokens"],
            "output_tokens":   coder_result["output_tokens"],
            "total_tokens":    coder_result["total_tokens"],
            "latency_sec":     coder_latency,
            "code":            generated_code if save_code else None,
            "exec_ok":         attempt_record.exec_ok,
            "test_pass":       attempt_record.test_pass,
            "status":          current_status,
            "error_type":      attempt_record.error_type,
            "error_stage":     attempt_record.error_stage,
            "error_message":   attempt_record.error_message,
            "tests_passed":    attempt_record.tests_passed,
            "tests_total":     attempt_record.tests_total,
            "code_length":     len(generated_code) if generated_code else 0,
            "selected":        None,
            "selection_rank":  None,
        }
        if hasattr(sample, "entry_point"):
            coder_step_entry["entry_point"] = sample.entry_point

        if save_step_level:
            append_jsonl(coder_step_entry, step_log_path)
            written_steps += 1

        global_step_id += 1

        pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
        print(f"  code (attempt 0): {pretty}")

        # ── 4-3. Repair loop ───────────────────────────
        final_attempt_record = attempt_record
        final_exec_result    = exec_result

        for repair_idx in range(max_repair):
            if attempt_record.passed:
                break

            previous_code  = generated_code
            repair_prompt  = build_repair_prompt_for_sample(
                sample,
                planner_output=planner_output,
                previous_code=previous_code,
                error_message=attempt_record.error_message,
            )

            t0 = time.perf_counter()
            repair_result  = _gen(repairer_model, repair_prompt, repairer_tokens)
            repair_latency = time.perf_counter() - t0

            repair_raw_text  = repair_result["text"]
            generated_code   = adapter.extract_code(sample, repair_raw_text)
            exec_result      = adapter.execute(sample, generated_code)
            attempt_record   = adapter.make_attempt_record(
                sample=sample,
                method=method_name,
                model_name=repairer_name,
                attempt_idx=repair_idx + 1,
                prompt=repair_prompt,
                raw_output=repair_raw_text,
                generated_code=generated_code,
                latency_sec=repair_latency,
                exec_result=exec_result,
            )

            call_count += 1
            cumulative_total_tokens += repair_result["total_tokens"]
            cumulative_latency      += repair_latency

            current_status = attempt_record.status
            transition_path.append(current_status)
            if str(current_status).startswith("EXEC_FAIL"):
                num_exec_fail += 1
            if str(current_status).startswith("TEST_FAIL"):
                num_test_fail += 1

            final_attempt_record = attempt_record
            final_exec_result    = exec_result

            repair_step_entry = {
                "run_id":          run_id,
                "dataset":         dataset_name,
                "problem_id":      problem_id,
                "method":          method_name,
                "trajectory_id":   trajectory_id,
                "step_id":         global_step_id,
                "call_index":      2 + repair_idx,
                "candidate_id":    0,
                "stage":           "repair",
                "is_retry":        False,
                "is_repair":       True,
                "is_planner":      False,
                "input_tokens":    repair_result["input_tokens"],
                "output_tokens":   repair_result["output_tokens"],
                "total_tokens":    repair_result["total_tokens"],
                "latency_sec":     repair_latency,
                "code":            generated_code if save_code else None,
                "exec_ok":         attempt_record.exec_ok,
                "test_pass":       attempt_record.test_pass,
                "status":          current_status,
                "error_type":      attempt_record.error_type,
                "error_stage":     attempt_record.error_stage,
                "error_message":   attempt_record.error_message,
                "tests_passed":    attempt_record.tests_passed,
                "tests_total":     attempt_record.tests_total,
                "code_length":     len(generated_code) if generated_code else 0,
                "selected":        None,
                "selection_rank":  None,
            }
            if hasattr(sample, "entry_point"):
                repair_step_entry["entry_point"] = sample.entry_point

            if save_step_level:
                append_jsonl(repair_step_entry, step_log_path)
                written_steps += 1

            global_step_id += 1

            pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
            print(f"  repair attempt {repair_idx + 1}: {pretty}")

            del repair_step_entry, repair_result, repair_raw_text

        # ── 4-4. 문제 종료 처리 ────────────────────────
        eval_results.append(final_exec_result)

        trajectory_entry = {
            "run_id":             run_id,
            "dataset":            dataset_name,
            "problem_id":         problem_id,
            "method":             method_name,
            "trajectory_id":      trajectory_id,
            "num_steps":          call_count,
            "call_count":         call_count,
            "final_status":       final_attempt_record.status,
            "final_tests_passed": final_attempt_record.tests_passed,
            "final_tests_total":  final_attempt_record.tests_total,
            "total_tokens":       cumulative_total_tokens,
            "total_latency":      cumulative_latency,
            "num_exec_fail":      num_exec_fail,
            "num_test_fail":      num_test_fail,
            "transition_path":    transition_path,
            "budget_used": {
                "tokens":  cumulative_total_tokens,
                "calls":   call_count,
                "latency": cumulative_latency,
            },
        }

        if save_trajectory_level:
            append_jsonl(trajectory_entry, trajectory_log_path)
            written_trajectories += 1

        sum_tokens  += trajectory_entry["total_tokens"]
        sum_latency += trajectory_entry["total_latency"]
        sum_calls   += trajectory_entry["call_count"]

        # transition 집계
        path = trajectory_entry["transition_path"]
        for j in range(len(path) - 1):
            coarse_a = path[j].split(":")[0]
            coarse_b = path[j + 1].split(":")[0]
            key = f"{coarse_a}->{coarse_b}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

        final_status = final_attempt_record.status
        if final_status not in ("PASS", "PLAN_DONE"):
            failure_type_counts[final_status] = failure_type_counts.get(final_status, 0) + 1

        # OOM 방지
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del planner_step_entry, coder_step_entry, trajectory_entry
        del final_attempt_record, final_exec_result, generated_code
        del planner_output, planner_raw_text, coder_raw_text
        del attempt_record, exec_result
        gc.collect()

    # ── 5. 결과 요약 ──────────────────────────────────
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
        print(f"  code_failed:      {extra_summary['code_failed']}")
        print(f"  setup_failed:     {extra_summary['setup_failed']}")
        print(f"  test_failed:      {extra_summary['test_failed']}")
        print(f"  semantic_failed:  {extra_summary['semantic_failed']}")
        print(f"  execution_failed: {extra_summary['execution_failed']}")
        print(f"{'=' * 60}")

    # ── 6. Problem-level summary ──
    n = len(eval_results)
    avg_tokens  = sum_tokens  / n if n else 0.0
    avg_latency = sum_latency / n if n else 0.0
    avg_calls   = sum_calls   / n if n else 0.0

    problem_summary = {
        "run_id":                  run_id,
        "dataset":                 dataset_name,
        "method":                  method_name,
        "total_problems":          summary["total"],
        "num_pass":                summary["passed"],
        "pass_at_1":               summary["pass@1"],
        "execution_success_rate":  summary["execution_success_rate"],
        "conditional_pass":        summary["conditional_pass"],
        "avg_tokens":              avg_tokens,
        "avg_latency":             avg_latency,
        "avg_calls":               avg_calls,
        "extra_summary":           extra_summary,
    }

    run_analysis = {
        "run_id":               run_id,
        "dataset":              dataset_name,
        "method":               method_name,
        "transition_counts":    transition_counts,
        "failure_type_counts":  failure_type_counts,
    }

    # ── 7. 저장 ──
    if save_step_level:
        print(f"💾 step_logs    : {step_log_path} ({written_steps}건)")
    if save_trajectory_level:
        print(f"💾 trajectory   : {trajectory_log_path} ({written_trajectories}건)")
    if save_problem_summary:
        save_result(problem_summary, os.path.join(output_dir, "summary.json"))
    if save_run_analysis:
        save_result(run_analysis, os.path.join(output_dir, "analysis.json"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.planner_coder_repair <config.yaml>")
        sys.exit(1)

    run_planner_coder_repair(sys.argv[1])
