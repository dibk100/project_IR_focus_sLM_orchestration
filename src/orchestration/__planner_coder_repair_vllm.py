"""
토큰을 나눠서 처리하는 것이 아닌 호출을 나눠서 처리하는 구조.

1. Planner
(input) -> plan출력

2. Coder
(input + plan) -> 출력

3. Repair
(문제 + plan + 이전 코드 + 에러 메시지) -> 출력 

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

from src.utils.io import save_result, append_jsonl
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


def make_step_entry(
    *,
    run_id: str,
    dataset_name: str,
    problem_id: str,
    method_name: str,
    trajectory_id: str,
    step_id: int,
    call_index: int,
    stage: str,
    gen_result: dict,
    latency_sec: float,
    code: str = None,
    attempt_record=None,
    planner_text: str = None,
    save_code: bool = True,
    sample=None,
):
    is_planner = stage == "plan"
    is_repair = stage == "repair"

    entry = {
        "run_id": run_id,
        "dataset": dataset_name,
        "problem_id": problem_id,
        "method": method_name,
        "trajectory_id": trajectory_id,
        "step_id": step_id,
        "call_index": call_index,
        "candidate_id": 0,
        "stage": stage,
        "is_retry": False,
        "is_repair": is_repair,
        "is_planner": is_planner,
        "input_tokens": gen_result["input_tokens"],
        "output_tokens": gen_result["output_tokens"],
        "total_tokens": gen_result["total_tokens"],
        "latency_sec": latency_sec,
        "code": (code if save_code else None) if not is_planner else None,
        "exec_ok": None if is_planner else attempt_record.exec_ok,
        "test_pass": None if is_planner else attempt_record.test_pass,
        "status": "PLAN_DONE" if is_planner else attempt_record.status,
        "error_type": None if is_planner else attempt_record.error_type,
        "error_stage": None if is_planner else attempt_record.error_stage,
        "error_message": None if is_planner else attempt_record.error_message,
        "tests_passed": None if is_planner else attempt_record.tests_passed,
        "tests_total": None if is_planner else attempt_record.tests_total,
        "code_length": 0 if is_planner else (len(code) if code else 0),
        "selected": None,
        "selection_rank": None,
    }

    if planner_text is not None:
        entry["plan_text"] = planner_text

    if hasattr(sample, "entry_point"):
        entry["entry_point"] = sample.entry_point

    return entry


def _gen(model: HFModel, prompt: str, max_tokens: int) -> dict:
    """
    HFModel.generate() 호출을 모든 stage에서 일관되게 처리하기 위한 wrapper.
    """
    orig = model.max_new_tokens
    model.max_new_tokens = max_tokens
    try:
        result = model.generate(prompt)
    finally:
        model.max_new_tokens = orig
    return result


def run_codegen_attempt(
    *,
    sample,
    model: HFModel,
    adapter,
    prompt: str,
    max_tokens: int,
    stage: str,  # "generate" | "repair"
    method_name: str,
    model_name: str,
    attempt_idx: int,
    use_planner_extraction: bool,
):
    """
    code-producing step(generate / repair)를 하나의 실행 함수로 통일.
    """
    t0 = time.perf_counter()
    gen_result = _gen(model, prompt, max_tokens)
    latency = time.perf_counter() - t0

    raw_text = gen_result["text"]

    if use_planner_extraction:
        generated_code = adapter.extract_code_for_planner(sample, raw_text)
    else:
        generated_code = adapter.extract_code(sample, raw_text)

    exec_result = adapter.execute(sample, generated_code)
    attempt_record = adapter.make_attempt_record(
        sample=sample,
        method=method_name,
        model_name=model_name,
        attempt_idx=attempt_idx,
        prompt=prompt,
        raw_output=raw_text,
        generated_code=generated_code,
        latency_sec=latency,
        exec_result=exec_result,
    )

    return {
        "stage": stage,
        "gen_result": gen_result,
        "raw_text": raw_text,
        "generated_code": generated_code,
        "exec_result": exec_result,
        "attempt_record": attempt_record,
        "latency": latency,
    }


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def run_planner_coder_repair(config_path: str):
    """Planner-Coder-Repair 실험 실행"""

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

    # nested model config (planner / coder / repairer)
    planner_cfg = model_cfg.get("planner", {})
    coder_cfg = model_cfg.get("coder", {})
    repairer_cfg = model_cfg.get("repairer", coder_cfg)

    planner_name = planner_cfg.get("name", model_cfg.get("name"))
    planner_tokens = planner_cfg.get("max_new_tokens", 256)
    planner_temp = planner_cfg.get("temperature", model_cfg.get("temperature", 0.2))

    coder_name = coder_cfg.get("name", model_cfg.get("name"))
    coder_tokens = coder_cfg.get("max_new_tokens", 512)
    coder_temp = coder_cfg.get("temperature", model_cfg.get("temperature", 0.2))

    repairer_name = repairer_cfg.get("name", coder_name)
    repairer_tokens = repairer_cfg.get("max_new_tokens", coder_tokens)
    repairer_temp = repairer_cfg.get("temperature", coder_temp)

    method_name = method_cfg.get("name", "planner_coder_repair")
    max_repair = budget_cfg.get("max_repair", 2)
    max_calls = budget_cfg.get("max_calls", 2 + max_repair)

    output_dir = output_cfg.get(
        "dir",
        f"results/phase1_ver3/{dataset_name}/planner_coder_repair"
    )

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🔧 Planner-Coder-Repair 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"max_repair          : {max_repair}  (total calls: 2+{max_repair}={2+max_repair})")
    print(f"max_calls(config)   : {max_calls}")
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
    common_backend = model_cfg.get("backend", "hf")
    api_base = model_cfg.get("api_base")
    api_key = model_cfg.get("api_key", "EMPTY")

    planner_model = HFModel(
        model_name=planner_name,
        max_new_tokens=planner_tokens,
        temperature=planner_temp,
        backend=common_backend,
        api_base=api_base,
        api_key=api_key,
    )

    coder_model = HFModel(
        model_name=coder_name,
        max_new_tokens=coder_tokens,
        temperature=coder_temp,
        backend=common_backend,
        api_base=api_base,
        api_key=api_key,
    )

    repairer_model = HFModel(
        model_name=repairer_name,
        max_new_tokens=repairer_tokens,
        temperature=repairer_temp,
        backend=common_backend,
        api_base=api_base,
        api_key=api_key,
    )

    print("✅ 모델 로딩 완료")

    # ── 4. 실험 루프 ──
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
    global_step_id = 0

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        problem_id = sample.task_id
        trajectory_id = f"{problem_id}_run0"

        print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

        cumulative_total_tokens = 0
        cumulative_latency = 0.0
        transition_path = []
        num_exec_fail = 0
        num_test_fail = 0
        call_count = 0

        # ── 4-1. Planner ───────────────────────────────
        planner_prompt = build_planner_prompt_for_sample(sample)

        planner_t0 = time.perf_counter()
        planner_gen_result = _gen(planner_model, planner_prompt, planner_tokens)
        planner_latency = time.perf_counter() - planner_t0

        planner_raw_text = planner_gen_result["text"]
        planner_output = extract_planner_output(planner_raw_text)

        call_count += 1
        cumulative_total_tokens += planner_gen_result["total_tokens"]
        cumulative_latency += planner_latency
        transition_path.append("PLAN_DONE")

        planner_step_entry = make_step_entry(
            run_id=run_id,
            dataset_name=dataset_name,
            problem_id=problem_id,
            method_name=method_name,
            trajectory_id=trajectory_id,
            step_id=global_step_id,
            call_index=0,
            stage="plan",
            gen_result=planner_gen_result,
            latency_sec=planner_latency,
            planner_text=planner_output,
            save_code=save_code,
            sample=sample,
        )

        if save_step_level:
            append_jsonl(planner_step_entry, step_log_path)
            written_steps += 1

        global_step_id += 1
        print(f"  📝 Plan: {planner_output[:80].replace(chr(10), ' ')}...")

        # ── 4-2. Initial Generate ──────────────────────
        coder_prompt = build_coder_prompt_for_sample(sample, planner_output)

        attempt = run_codegen_attempt(
            sample=sample,
            model=coder_model,
            adapter=adapter,
            prompt=coder_prompt,
            max_tokens=coder_tokens,
            stage="generate",
            method_name=method_name,
            model_name=coder_name,
            attempt_idx=0,
            use_planner_extraction=True,
        )

        call_count += 1
        cumulative_total_tokens += attempt["gen_result"]["total_tokens"]
        cumulative_latency += attempt["latency"]

        current_status = attempt["attempt_record"].status
        transition_path.append(current_status)

        if str(current_status).startswith("EXEC_FAIL"):
            num_exec_fail += 1
        if str(current_status).startswith("TEST_FAIL"):
            num_test_fail += 1

        coder_step_entry = make_step_entry(
            run_id=run_id,
            dataset_name=dataset_name,
            problem_id=problem_id,
            method_name=method_name,
            trajectory_id=trajectory_id,
            step_id=global_step_id,
            call_index=1,
            stage="generate",
            gen_result=attempt["gen_result"],
            latency_sec=attempt["latency"],
            code=attempt["generated_code"],
            attempt_record=attempt["attempt_record"],
            save_code=save_code,
            sample=sample,
        )

        if save_step_level:
            append_jsonl(coder_step_entry, step_log_path)
            written_steps += 1

        global_step_id += 1

        pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
        print(f"  code (attempt 0): {pretty}")

        # ── 4-3. Repair loop ───────────────────────────
        repair_idx = 0
        while (not attempt["attempt_record"].passed) and (repair_idx < max_repair):
            repair_prompt = build_repair_prompt_for_sample(
                sample,
                planner_output=planner_output,
                previous_code=attempt["generated_code"],
                error_message=attempt["attempt_record"].error_message,
            )

            attempt = run_codegen_attempt(
                sample=sample,
                model=repairer_model,
                adapter=adapter,
                prompt=repair_prompt,
                max_tokens=repairer_tokens,
                stage="repair",
                method_name=method_name,
                model_name=repairer_name,
                attempt_idx=repair_idx + 1,
                use_planner_extraction=False,
            )

            call_count += 1
            cumulative_total_tokens += attempt["gen_result"]["total_tokens"]
            cumulative_latency += attempt["latency"]

            current_status = attempt["attempt_record"].status
            transition_path.append(current_status)

            if str(current_status).startswith("EXEC_FAIL"):
                num_exec_fail += 1
            if str(current_status).startswith("TEST_FAIL"):
                num_test_fail += 1

            repair_step_entry = make_step_entry(
                run_id=run_id,
                dataset_name=dataset_name,
                problem_id=problem_id,
                method_name=method_name,
                trajectory_id=trajectory_id,
                step_id=global_step_id,
                call_index=2 + repair_idx,
                stage="repair",
                gen_result=attempt["gen_result"],
                latency_sec=attempt["latency"],
                code=attempt["generated_code"],
                attempt_record=attempt["attempt_record"],
                save_code=save_code,
                sample=sample,
            )

            if save_step_level:
                append_jsonl(repair_step_entry, step_log_path)
                written_steps += 1

            global_step_id += 1

            pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
            print(f"  repair attempt {repair_idx + 1}: {pretty}")

            del repair_step_entry
            repair_idx += 1

        # ── 4-4. 문제 종료 처리 ────────────────────────
        final_attempt_record = attempt["attempt_record"]
        final_exec_result = attempt["exec_result"]

        eval_results.append(final_exec_result)

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

        sum_tokens += trajectory_entry["total_tokens"]
        sum_latency += trajectory_entry["total_latency"]
        sum_calls += trajectory_entry["call_count"]

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
        del final_attempt_record, final_exec_result
        del planner_output, planner_raw_text
        del attempt
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

    run_analysis = {
        "run_id": run_id,
        "dataset": dataset_name,
        "method": method_name,
        "transition_counts": transition_counts,
        "failure_type_counts": failure_type_counts,
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