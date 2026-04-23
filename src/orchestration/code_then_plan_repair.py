"""
Code-Then-Plan-Repair

Problem
  -> G(Generate) -> [[실패 시] D(Plan) -> G(Plan-Code) -> [실패 시] R(Repair)-> G(Generate)] x K 반복
  
1. 초기 직접 코드 생성 (single_shot과 동일)                  [action="generate"]
2. 실행/평가
3. 성공하면 종료 (1 call)
4. 실패 시:
   a. Planner가 구현 계획 생성                               [action="plan"]
   b. Coder가 계획 기반 코드 재생성                           [action="generate"]
   c. 실행/평가
5. 여전히 실패 시:
   a. Repairer가 실패 코드 + 에러 메시지 기반 수정             [action="repair"]
   b. 실행/평가
6. 결과 저장 (최소 4 calls)

- 첫 시도는 single_shot과 동일
- 실패 시 planning 기반 복구
- plan으로도 실패하면 마지막으로 repair
- 이후 planner_output까지 repair prompt에 넣는 k번 방식으로 확장 가능
    - if max_call = 21 : 1 (initial) + {2 (plan+code) + 1 (repair)}*6회까지 반복 가능
"""
from __future__ import annotations

import gc
import os
import sys
import time
from types import SimpleNamespace

import yaml
import torch

from src.models.hf_model import HFModel
from src.evaluation.metrics import summarize_failure_breakdown, summarize_phase1_results
from src.utils.io import save_result, save_results_jsonl, make_run_id
from src.utils.dataloader import load_task_and_adapter
from src.utils.prompt_loader import (
    build_planner_prompt_for_sample,
    build_coder_prompt_for_sample,
    build_repair_prompt_for_sample,
)
from src.utils.prompting.planner_coder import extract_planner_output


# ─────────────────────────────────────────────
# Debug helpers
# ─────────────────────────────────────────────

def _shorten(text, max_len: int = 500) -> str:
    if text is None:
        return "(None)"
    s = str(text)
    return s if len(s) <= max_len else s[:max_len] + f"\n... (truncated, total {len(s)} chars)"


def _print_header(title: str, width: int = 60):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print("=" * width)


def _print_sample_execution_flow(sample, prompt: str, raw_text: str, generated_code: str | None = None):
    _print_header("SAMPLE DEBUG :: STEP 1. INPUT PROMPT")
    print(_shorten(prompt))

    _print_header("SAMPLE DEBUG :: STEP 2. RAW MODEL OUTPUT (y)")
    print(_shorten(raw_text))

    _print_header("SAMPLE DEBUG :: STEP 3. EXECUTABLE CODE")
    if generated_code is None:
        print("(empty or not yet extracted — skipping execution steps)")
        return
    print(_shorten(generated_code))

    if hasattr(sample, "test") and hasattr(sample, "entry_point"):
        namespace = {}

        _print_header("SAMPLE DEBUG :: STEP 4. exec(code) INPUT")
        print(_shorten(generated_code))

        try:
            exec(generated_code, namespace)
            print("\n[exec(code) SUCCESS]")
            print("namespace keys:", list(namespace.keys()))
        except Exception as e:
            print("\n[exec(code) FAILED]")
            print(repr(e))
            return

        candidate = namespace.get(sample.entry_point, None)

        _print_header("SAMPLE DEBUG :: STEP 4-1. candidate")
        print(f"entry_point = {sample.entry_point}")
        print("candidate =", repr(candidate))

        _print_header("SAMPLE DEBUG :: STEP 5. exec(test) INPUT")
        print(_shorten(sample.test))

        try:
            exec(sample.test, namespace)
            print("\n[exec(test) SUCCESS]")
            print("namespace keys:", list(namespace.keys()))
        except Exception as e:
            print("\n[exec(test) FAILED]")
            print(repr(e))
            return

        check_fn = namespace.get("check", None)

        _print_header("SAMPLE DEBUG :: STEP 5-1. check function")
        print("check =", repr(check_fn))

        _print_header("SAMPLE DEBUG :: STEP 6. check(candidate)")
        try:
            check_fn(candidate)
            print("PASS")
        except Exception as e:
            print("FAILED")
            print(repr(e))

    elif hasattr(sample, "test_list"):
        _print_header("SAMPLE DEBUG :: MBPP TEST LIST")
        for idx, test_case in enumerate(sample.test_list):
            print(f"[test_list[{idx}]]")
            print(_shorten(test_case))
            print("-" * 40)

    elif hasattr(sample, "test"):
        _print_header("SAMPLE DEBUG :: TEST CODE")
        print(_shorten(sample.test))


def _collect_failure_example(
    failure_examples: dict,
    *,
    problem_id: str,
    attempt_idx: int,
    status: str,
    prompt: str,
    raw_text: str,
    generated_code: str | None,
    error_type: str | None,
    error_stage: str | None,
    error_message: str | None,
):
    if status == "PASS" or status in failure_examples:
        return

    failure_examples[status] = {
        "problem_id": problem_id,
        "attempt_idx": attempt_idx,
        "status": status,
        "error_type": error_type,
        "error_stage": error_stage,
        "error_message": _shorten(error_message, max_len=2000),
        "prompt": _shorten(prompt, max_len=2000),
        "raw_text": _shorten(raw_text, max_len=2000),
        "generated_code": _shorten(generated_code, max_len=2000),
    }


def _make_empty_output_record(message: str):
    return SimpleNamespace(
        status="CODE_FAIL:empty_output",
        tests_passed=0,
        tests_total=None,
        passed=False,
        exec_ok=False,
        test_pass=False,
        error_type="empty_output",
        error_stage="code",
        error_message=message,
    )


def _extract_code_for_planner(adapter, sample, raw_text: str):
    if hasattr(adapter, "extract_code_for_planner"):
        return adapter.extract_code_for_planner(sample, raw_text)
    return adapter.extract_code(sample, raw_text)


def _extract_code_for_repair(adapter, sample, raw_text: str):
    if hasattr(adapter, "extract_code_for_repair"):
        return adapter.extract_code_for_repair(sample, raw_text)
    if hasattr(adapter, "extract_code_for_planner"):
        return adapter.extract_code_for_planner(sample, raw_text)
    return adapter.extract_code(sample, raw_text)


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def run_code_then_plan_repair(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg = config.get("run", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    method_cfg = config.get("method", {})
    budget_cfg = config.get("budget", {})
    output_cfg = config.get("output", {})
    logging_cfg = config.get("logging", {})
    debug_cfg = config.get("debug", {})

    run_id = make_run_id(config)
    seed = run_cfg.get("seed", 42)

    dataset_name = dataset_cfg.get("name", "humaneval")
    num_samples = dataset_cfg.get("num_samples", 1)

    model_name = model_cfg.get("name", "")
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    temperature = model_cfg.get("temperature", 0.0)

    planner_cfg_inner = model_cfg.get("planner", {})
    planner_max_tokens = planner_cfg_inner.get("max_new_tokens", 256)

    coder_cfg_inner = model_cfg.get("coder", {})
    coder_max_tokens = coder_cfg_inner.get("max_new_tokens", max_new_tokens)

    repair_cfg_inner = model_cfg.get("repair", {})
    repair_max_tokens = repair_cfg_inner.get("max_new_tokens", max_new_tokens)

    method_name = method_cfg.get("name", "code_then_plan_repair")
    max_calls = budget_cfg.get("max_calls", 21)

    output_dir = output_cfg.get("dir", f"results/RUN/{dataset_name}/code_then_plan_repair")

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    debug_mode = debug_cfg.get("mode", "run")
 
    os.makedirs(output_dir, exist_ok=True)

    log_fp = None
    stdout_backup = sys.stdout
    debug_log_path = os.path.join(output_dir, "sample_debug_output.txt")

    try:
        if debug_mode == "sample":
            log_fp = open(debug_log_path, "w", encoding="utf-8")
            sys.stdout = log_fp

        print("=" * 60)
        print("🔄 Code-Then-Plan-Repair 실험")
        print("=" * 60)
        print(f"run_id              : {run_id}")
        print(f"dataset             : {dataset_name}")
        print(f"method              : {method_name}")
        print(f"model               : {model_name}")
        print(f"max_new_tokens      : {max_new_tokens}")
        print(f"planner_max_tokens  : {planner_max_tokens}")
        print(f"coder_max_tokens    : {coder_max_tokens}")
        print(f"repair_max_tokens   : {repair_max_tokens}")
        print(f"max_calls           : {max_calls}")
        print(f"seed                : {seed}")
        print(f"debug_mode          : {debug_mode}")
        print(f"output_dir          : {output_dir}")
        print("=" * 60)

        save_result(
            {
                "run": {"run_id": run_id, "seed": seed},
                "dataset": dataset_cfg,
                "model": model_cfg,
                "method": method_cfg,
                "budget": budget_cfg,
                "output": output_cfg,
                "logging": logging_cfg,
                "debug": debug_cfg,
                "config_path": config_path,
            },
            os.path.join(output_dir, "config.json"),
        )

        task, adapter = load_task_and_adapter(dataset_name)
        print(f"📦 데이터셋: {dataset_name} | size={len(task)}")

        print(f"🔄 모델 로딩: {model_name}")
        model = HFModel(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print("✅ 모델 로딩 완료")

        step_logs = []
        trajectory_logs = []
        eval_results = []
        failure_examples = {}

        samples_to_run = min(num_samples, len(task))

        for i in range(samples_to_run):
            sample = task.get_sample(i)
            problem_id = sample.task_id
            trajectory_id = f"{problem_id}_run0"

            print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

            cumulative_input_tokens = 0
            cumulative_output_tokens = 0
            cumulative_total_tokens = 0
            cumulative_latency = 0.0
            transition_path = []
            num_exec_fail = 0
            num_test_fail = 0
            call_count = 0

            final_exec_result = None
            final_attempt_record = None
            used_plan = False
            used_repair = False
            latest_plan = None
            latest_code = None
            planning_cycle_count = 0

            # =========================================================
            # Stage 1: Direct generation
            # =========================================================
            if call_count < max_calls:
                initial_prompt = adapter.build_initial_prompt(sample)

                gen_start = time.perf_counter()
                gen_result = model.generate(initial_prompt)
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

                if output_tokens == 0 or not raw_text.strip():
                    final_attempt_record = _make_empty_output_record(
                        "Model output was empty or contained no extractable code."
                    )
                    final_exec_result = final_attempt_record
                    current_status = final_attempt_record.status
                    transition_path.append(current_status)

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
                        "policy_action": "generate",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "latency_sec": latency_sec,
                        "code": None,
                        "planner_output": None,
                        "exec_ok": False,
                        "test_pass": False,
                        "status": current_status,
                        "error_type": final_attempt_record.error_type,
                        "error_stage": final_attempt_record.error_stage,
                        "error_message": final_attempt_record.error_message,
                        "tests_passed": final_attempt_record.tests_passed,
                        "tests_total": final_attempt_record.tests_total,
                        "code_length": 0,
                        "selected": None,
                        "selection_rank": None,
                    }
                    if hasattr(sample, "entry_point"):
                        step_entry["entry_point"] = sample.entry_point

                    step_logs.append(step_entry)
                    print(f"  generate: ❌ {current_status}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=0,
                        status=current_status,
                        prompt=initial_prompt,
                        raw_text=raw_text,
                        generated_code=None,
                        error_type=final_attempt_record.error_type,
                        error_stage=final_attempt_record.error_stage,
                        error_message=final_attempt_record.error_message,
                    )

                else:
                    generated_code = adapter.extract_code(sample, raw_text)

                    if debug_mode == "sample":
                        _print_sample_execution_flow(
                            sample=sample,
                            prompt=initial_prompt,
                            raw_text=raw_text,
                            generated_code=generated_code,
                        )

                    exec_result = adapter.execute(sample, generated_code)

                    if debug_mode == "sample":
                        _print_header("SAMPLE DEBUG :: EXEC RESULT OBJECT")
                        print(exec_result)
                        _print_header("SAMPLE DEBUG :: CLASSIFIED EXECUTION")
                        print(adapter.classify_execution(exec_result))

                    attempt_record = adapter.make_attempt_record(
                        sample=sample,
                        method=method_name,
                        model_name=model_name,
                        attempt_idx=0,
                        prompt=initial_prompt,
                        raw_output=raw_text,
                        generated_code=generated_code,
                        latency_sec=latency_sec,
                        exec_result=exec_result,
                    )

                    if debug_mode == "sample":
                        _print_header("SAMPLE DEBUG :: ATTEMPT RECORD")
                        print(attempt_record)

                    final_attempt_record = attempt_record
                    final_exec_result = exec_result
                    latest_code = generated_code
                    current_status = attempt_record.status
                    transition_path.append(current_status)

                    if str(current_status).startswith("EXEC_FAIL"):
                        num_exec_fail += 1
                    if str(current_status).startswith("TEST_FAIL"):
                        num_test_fail += 1

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
                        "policy_action": "generate",
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "latency_sec": latency_sec,
                        "code": generated_code if save_code else None,
                        "planner_output": None,
                        "exec_ok": attempt_record.exec_ok,
                        "test_pass": attempt_record.test_pass,
                        "status": current_status,
                        "error_type": attempt_record.error_type,
                        "error_stage": attempt_record.error_stage,
                        "error_message": attempt_record.error_message,
                        "tests_passed": attempt_record.tests_passed,
                        "tests_total": attempt_record.tests_total,
                        "code_length": len(generated_code) if generated_code else 0,
                        "selected": None,
                        "selection_rank": None,
                    }
                    if hasattr(sample, "entry_point"):
                        step_entry["entry_point"] = sample.entry_point

                    step_logs.append(step_entry)

                    pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
                    print(f"  generate: {pretty}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=0,
                        status=current_status,
                        prompt=initial_prompt,
                        raw_text=raw_text,
                        generated_code=generated_code,
                        error_type=attempt_record.error_type,
                        error_stage=attempt_record.error_stage,
                        error_message=attempt_record.error_message,
                    )

            # =========================================================
            # Planning + repair cycle
            # Each cycle:
            #   plan -> plan_code -> (if fail) repair
            # =========================================================
            while (
                final_attempt_record is not None
                and not final_attempt_record.passed
                and call_count + 2 <= max_calls
            ):
                planning_cycle_count += 1
                used_plan = True

                # ----------------------------
                # 1) plan
                # ----------------------------
                planner_prompt = build_planner_prompt_for_sample(sample)

                orig_tokens = model.max_new_tokens
                model.max_new_tokens = planner_max_tokens
                try:
                    plan_start = time.perf_counter()
                    plan_gen_result = model.generate(planner_prompt)
                    plan_end = time.perf_counter()
                finally:
                    model.max_new_tokens = orig_tokens

                plan_latency = plan_end - plan_start
                plan_raw_text = plan_gen_result["text"]
                plan_input_tokens = plan_gen_result["input_tokens"]
                plan_output_tokens = plan_gen_result["output_tokens"]
                plan_total_tokens = plan_gen_result["total_tokens"]

                call_count += 1
                cumulative_input_tokens += plan_input_tokens
                cumulative_output_tokens += plan_output_tokens
                cumulative_total_tokens += plan_total_tokens
                cumulative_latency += plan_latency

                latest_plan = extract_planner_output(plan_raw_text)
                transition_path.append("PLAN_DONE")

                plan_step_entry = {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "problem_id": problem_id,
                    "method": method_name,
                    "trajectory_id": trajectory_id,
                    "step_id": len(step_logs),
                    "call_index": call_count - 1,
                    "candidate_id": 0,
                    "stage": "plan",
                    "is_retry": False,
                    "is_repair": False,
                    "is_planner": True,
                    "policy_action": "plan",
                    "input_tokens": plan_input_tokens,
                    "output_tokens": plan_output_tokens,
                    "total_tokens": plan_total_tokens,
                    "latency_sec": plan_latency,
                    "code": None,
                    "planner_output": latest_plan if save_code else None,
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
                    plan_step_entry["entry_point"] = sample.entry_point

                step_logs.append(plan_step_entry)
                print(f"  plan[{planning_cycle_count}]: 📝 {latest_plan[:80].replace(chr(10), ' ')}...")

                # ----------------------------
                # 2) plan_code
                # ----------------------------
                coder_prompt = build_coder_prompt_for_sample(sample, latest_plan)

                model.max_new_tokens = coder_max_tokens
                try:
                    code_start = time.perf_counter()
                    code_gen_result = model.generate(coder_prompt)
                    code_end = time.perf_counter()
                finally:
                    model.max_new_tokens = orig_tokens

                code_latency = code_end - code_start
                code_raw_text = code_gen_result["text"]
                code_input_tokens = code_gen_result["input_tokens"]
                code_output_tokens = code_gen_result["output_tokens"]
                code_total_tokens = code_gen_result["total_tokens"]

                call_count += 1
                cumulative_input_tokens += code_input_tokens
                cumulative_output_tokens += code_output_tokens
                cumulative_total_tokens += code_total_tokens
                cumulative_latency += code_latency

                if code_output_tokens == 0 or not code_raw_text.strip():
                    final_attempt_record = _make_empty_output_record(
                        "Model output was empty (plan_code stage)."
                    )
                    final_exec_result = final_attempt_record
                    latest_code = None
                    current_status = final_attempt_record.status
                    transition_path.append(current_status)

                    code_step_entry = {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "problem_id": problem_id,
                        "method": method_name,
                        "trajectory_id": trajectory_id,
                        "step_id": len(step_logs),
                        "call_index": call_count - 1,
                        "candidate_id": 0,
                        "stage": "plan_code",
                        "is_retry": False,
                        "is_repair": False,
                        "is_planner": False,
                        "policy_action": "generate",
                        "input_tokens": code_input_tokens,
                        "output_tokens": code_output_tokens,
                        "total_tokens": code_total_tokens,
                        "latency_sec": code_latency,
                        "code": None,
                        "planner_output": None,
                        "exec_ok": False,
                        "test_pass": False,
                        "status": current_status,
                        "error_type": final_attempt_record.error_type,
                        "error_stage": final_attempt_record.error_stage,
                        "error_message": final_attempt_record.error_message,
                        "tests_passed": final_attempt_record.tests_passed,
                        "tests_total": final_attempt_record.tests_total,
                        "code_length": 0,
                        "selected": None,
                        "selection_rank": None,
                    }
                    if hasattr(sample, "entry_point"):
                        code_step_entry["entry_point"] = sample.entry_point

                    step_logs.append(code_step_entry)
                    print(f"  plan_code[{planning_cycle_count}]: ❌ {current_status}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=planning_cycle_count,
                        status=current_status,
                        prompt=coder_prompt,
                        raw_text=code_raw_text,
                        generated_code=None,
                        error_type=final_attempt_record.error_type,
                        error_stage=final_attempt_record.error_stage,
                        error_message=final_attempt_record.error_message,
                    )

                else:
                    code_generated = _extract_code_for_planner(adapter, sample, code_raw_text)

                    if debug_mode == "sample":
                        _print_sample_execution_flow(
                            sample=sample,
                            prompt=coder_prompt,
                            raw_text=code_raw_text,
                            generated_code=code_generated,
                        )

                    exec_result = adapter.execute(sample, code_generated)

                    if debug_mode == "sample":
                        _print_header("SAMPLE DEBUG :: EXEC RESULT OBJECT (plan_code)")
                        print(exec_result)
                        _print_header("SAMPLE DEBUG :: CLASSIFIED EXECUTION (plan_code)")
                        print(adapter.classify_execution(exec_result))

                    code_record = adapter.make_attempt_record(
                        sample=sample,
                        method=method_name,
                        model_name=model_name,
                        attempt_idx=planning_cycle_count,
                        prompt=coder_prompt,
                        raw_output=code_raw_text,
                        generated_code=code_generated,
                        latency_sec=code_latency,
                        exec_result=exec_result,
                    )

                    if debug_mode == "sample":
                        _print_header("SAMPLE DEBUG :: ATTEMPT RECORD (plan_code)")
                        print(code_record)

                    final_attempt_record = code_record
                    final_exec_result = exec_result
                    latest_code = code_generated
                    current_status = code_record.status
                    transition_path.append(current_status)

                    if str(current_status).startswith("EXEC_FAIL"):
                        num_exec_fail += 1
                    if str(current_status).startswith("TEST_FAIL"):
                        num_test_fail += 1

                    code_step_entry = {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "problem_id": problem_id,
                        "method": method_name,
                        "trajectory_id": trajectory_id,
                        "step_id": len(step_logs),
                        "call_index": call_count - 1,
                        "candidate_id": 0,
                        "stage": "plan_code",
                        "is_retry": False,
                        "is_repair": False,
                        "is_planner": False,
                        "policy_action": "generate",
                        "input_tokens": code_input_tokens,
                        "output_tokens": code_output_tokens,
                        "total_tokens": code_total_tokens,
                        "latency_sec": code_latency,
                        "code": code_generated if save_code else None,
                        "planner_output": None,
                        "exec_ok": code_record.exec_ok,
                        "test_pass": code_record.test_pass,
                        "status": current_status,
                        "error_type": code_record.error_type,
                        "error_stage": code_record.error_stage,
                        "error_message": code_record.error_message,
                        "tests_passed": code_record.tests_passed,
                        "tests_total": code_record.tests_total,
                        "code_length": len(code_generated) if code_generated else 0,
                        "selected": None,
                        "selection_rank": None,
                    }
                    if hasattr(sample, "entry_point"):
                        code_step_entry["entry_point"] = sample.entry_point

                    step_logs.append(code_step_entry)

                    pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
                    print(f"  plan_code[{planning_cycle_count}]: {pretty}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=planning_cycle_count,
                        status=current_status,
                        prompt=coder_prompt,
                        raw_text=code_raw_text,
                        generated_code=code_generated,
                        error_type=code_record.error_type,
                        error_stage=code_record.error_stage,
                        error_message=code_record.error_message,
                    )

                if final_attempt_record.passed:
                    break

                # repair budget check
                if call_count >= max_calls:
                    break

                # ----------------------------
                # 3) repair
                # ----------------------------
                used_repair = True

                repair_prompt = build_repair_prompt_for_sample(
                    sample=sample,
                    previous_code=latest_code,
                    error_message=getattr(final_attempt_record, "error_message", None),
                    failing_status=final_attempt_record.status,
                    planner_output=latest_plan,
                )

                model.max_new_tokens = repair_max_tokens
                try:
                    repair_start = time.perf_counter()
                    repair_gen_result = model.generate(repair_prompt)
                    repair_end = time.perf_counter()
                finally:
                    model.max_new_tokens = orig_tokens

                repair_latency = repair_end - repair_start
                repair_raw_text = repair_gen_result["text"]
                repair_input_tokens = repair_gen_result["input_tokens"]
                repair_output_tokens = repair_gen_result["output_tokens"]
                repair_total_tokens = repair_gen_result["total_tokens"]

                call_count += 1
                cumulative_input_tokens += repair_input_tokens
                cumulative_output_tokens += repair_output_tokens
                cumulative_total_tokens += repair_total_tokens
                cumulative_latency += repair_latency

                if repair_output_tokens == 0 or not repair_raw_text.strip():
                    final_attempt_record = _make_empty_output_record(
                        "Model output was empty (repair stage)."
                    )
                    final_exec_result = final_attempt_record
                    latest_code = None
                    current_status = final_attempt_record.status
                    transition_path.append(current_status)

                    repair_step_entry = {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "problem_id": problem_id,
                        "method": method_name,
                        "trajectory_id": trajectory_id,
                        "step_id": len(step_logs),
                        "call_index": call_count - 1,
                        "candidate_id": 0,
                        "stage": "repair",
                        "is_retry": False,
                        "is_repair": True,
                        "is_planner": False,
                        "policy_action": "repair",
                        "input_tokens": repair_input_tokens,
                        "output_tokens": repair_output_tokens,
                        "total_tokens": repair_total_tokens,
                        "latency_sec": repair_latency,
                        "code": None,
                        "planner_output": None,
                        "exec_ok": False,
                        "test_pass": False,
                        "status": current_status,
                        "error_type": final_attempt_record.error_type,
                        "error_stage": final_attempt_record.error_stage,
                        "error_message": final_attempt_record.error_message,
                        "tests_passed": final_attempt_record.tests_passed,
                        "tests_total": final_attempt_record.tests_total,
                        "code_length": 0,
                        "selected": None,
                        "selection_rank": None,
                    }
                    if hasattr(sample, "entry_point"):
                        repair_step_entry["entry_point"] = sample.entry_point

                    step_logs.append(repair_step_entry)
                    print(f"  repair[{planning_cycle_count}]: ❌ {current_status}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=planning_cycle_count,
                        status=current_status,
                        prompt=repair_prompt,
                        raw_text=repair_raw_text,
                        generated_code=None,
                        error_type=final_attempt_record.error_type,
                        error_stage=final_attempt_record.error_stage,
                        error_message=final_attempt_record.error_message,
                    )

                else:
                    repaired_code = _extract_code_for_repair(adapter, sample, repair_raw_text)

                    if debug_mode == "sample":
                        _print_sample_execution_flow(
                            sample=sample,
                            prompt=repair_prompt,
                            raw_text=repair_raw_text,
                            generated_code=repaired_code,
                        )

                    exec_result = adapter.execute(sample, repaired_code)

                    if debug_mode == "sample":
                        _print_header("SAMPLE DEBUG :: EXEC RESULT OBJECT (repair)")
                        print(exec_result)
                        _print_header("SAMPLE DEBUG :: CLASSIFIED EXECUTION (repair)")
                        print(adapter.classify_execution(exec_result))

                    repair_record = adapter.make_attempt_record(
                        sample=sample,
                        method=method_name,
                        model_name=model_name,
                        attempt_idx=planning_cycle_count,
                        prompt=repair_prompt,
                        raw_output=repair_raw_text,
                        generated_code=repaired_code,
                        latency_sec=repair_latency,
                        exec_result=exec_result,
                    )

                    if debug_mode == "sample":
                        _print_header("SAMPLE DEBUG :: ATTEMPT RECORD (repair)")
                        print(repair_record)

                    final_attempt_record = repair_record
                    final_exec_result = exec_result
                    latest_code = repaired_code
                    current_status = repair_record.status
                    transition_path.append(current_status)

                    if str(current_status).startswith("EXEC_FAIL"):
                        num_exec_fail += 1
                    if str(current_status).startswith("TEST_FAIL"):
                        num_test_fail += 1

                    repair_step_entry = {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "problem_id": problem_id,
                        "method": method_name,
                        "trajectory_id": trajectory_id,
                        "step_id": len(step_logs),
                        "call_index": call_count - 1,
                        "candidate_id": 0,
                        "stage": "repair",
                        "is_retry": False,
                        "is_repair": True,
                        "is_planner": False,
                        "policy_action": "repair",
                        "input_tokens": repair_input_tokens,
                        "output_tokens": repair_output_tokens,
                        "total_tokens": repair_total_tokens,
                        "latency_sec": repair_latency,
                        "code": repaired_code if save_code else None,
                        "planner_output": latest_plan if save_code else None,
                        "exec_ok": repair_record.exec_ok,
                        "test_pass": repair_record.test_pass,
                        "status": current_status,
                        "error_type": repair_record.error_type,
                        "error_stage": repair_record.error_stage,
                        "error_message": repair_record.error_message,
                        "tests_passed": repair_record.tests_passed,
                        "tests_total": repair_record.tests_total,
                        "code_length": len(repaired_code) if repaired_code else 0,
                        "selected": None,
                        "selection_rank": None,
                    }
                    if hasattr(sample, "entry_point"):
                        repair_step_entry["entry_point"] = sample.entry_point

                    step_logs.append(repair_step_entry)

                    pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
                    print(f"  repair[{planning_cycle_count}]: {pretty}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=planning_cycle_count,
                        status=current_status,
                        prompt=repair_prompt,
                        raw_text=repair_raw_text,
                        generated_code=repaired_code,
                        error_type=repair_record.error_type,
                        error_stage=repair_record.error_stage,
                        error_message=repair_record.error_message,
                    )

                if final_attempt_record.passed:
                    break

            if final_exec_result is None:
                final_exec_result = _make_empty_output_record("No executable result was produced.")
            if final_attempt_record is None:
                final_attempt_record = _make_empty_output_record("No attempt record was produced.")

            eval_results.append(final_exec_result)

            final_status = final_attempt_record.status
            failure_family = "PASS" if final_status == "PASS" else str(final_status).split(":")[0]

            trajectory_entry = {
                "run_id": run_id,
                "dataset": dataset_name,
                "problem_id": problem_id,
                "method": method_name,
                "trajectory_id": trajectory_id,
                "num_steps": call_count,
                "call_count": call_count,
                "final_status": final_status,
                "failure_family": failure_family,
                "final_tests_passed": final_attempt_record.tests_passed,
                "final_tests_total": final_attempt_record.tests_total,
                "total_input_tokens": cumulative_input_tokens,
                "total_output_tokens": cumulative_output_tokens,
                "total_tokens": cumulative_total_tokens,
                "total_latency": cumulative_latency,
                "num_exec_fail": num_exec_fail,
                "num_test_fail": num_test_fail,
                "transition_path": transition_path,
                "used_plan": used_plan,
                "used_repair": used_repair,
                "budget_used": {
                    "tokens": cumulative_total_tokens,
                    "calls": call_count,
                    "latency": cumulative_latency,
                },
                "recovered_by": (
                    "generate" if (not used_plan and final_status == "PASS")
                    else "plan_code" if (used_plan and not used_repair and final_status == "PASS")
                    else "repair" if (used_repair and final_status == "PASS")
                    else None
                ),
                "planning_recovery_attempt": (
                    planning_cycle_count if (used_plan and final_status == "PASS")
                    else None
                ),
            }

            trajectory_logs.append(trajectory_entry)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

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

        extra_summary = summarize_failure_breakdown(eval_results)
        print(f"  code_failed: {extra_summary['code_failed']}")
        print(f"  define_test_failed: {extra_summary['define_test_failed']}")
        print(f"  run_test_failed: {extra_summary['run_test_failed']}")
        print(f"{'=' * 60}")

        n_used_plan = sum(1 for t in trajectory_logs if t.get("used_plan", False))
        n_used_repair = sum(1 for t in trajectory_logs if t.get("used_repair", False))
        n_planning_recovered = sum(
            1 for t in trajectory_logs
            if t.get("recovered_by") in ("plan_code", "repair")
        )
        n_repair_recovered = sum(
            1 for t in trajectory_logs
            if t.get("recovered_by") == "repair"
        )

        print(
            f"  plan 사용: {n_used_plan}/{len(trajectory_logs)} ({n_used_plan/len(trajectory_logs)*100:.1f}%)"
            if len(trajectory_logs) > 0 else "  plan 사용: 0/0"
        )
        print(
            f"  repair 사용: {n_used_repair}/{len(trajectory_logs)} ({n_used_repair/len(trajectory_logs)*100:.1f}%)"
            if len(trajectory_logs) > 0 else "  repair 사용: 0/0"
        )
        print(
            f"  planning 복구 성공: {n_planning_recovered}/{n_used_plan} ({n_planning_recovered/n_used_plan*100:.1f}%)"
            if n_used_plan > 0 else "  planning 복구 성공: 0/0"
        )
        print(
            f"  repair 복구 성공: {n_repair_recovered}/{n_used_repair} ({n_repair_recovered/n_used_repair*100:.1f}%)"
            if n_used_repair > 0 else "  repair 복구 성공: 0/0"
        )
        print(f"{'=' * 60}")

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
            "planning_stats": {
                "used_plan": n_used_plan,
                "used_repair": n_used_repair,
                "planning_recovered": n_planning_recovered,
                "repair_recovered": n_repair_recovered,
                "planning_recovery_rate": n_planning_recovered / n_used_plan if n_used_plan > 0 else 0.0,
                "repair_recovery_rate": n_repair_recovered / n_used_repair if n_used_repair > 0 else 0.0,
            },
        }

        transition_counts = {}
        failure_type_counts = {}
        failure_family_counts = {}

        for traj in trajectory_logs:
            path = traj["transition_path"]
            for j in range(len(path) - 1):
                coarse_a = path[j].split(":")[0]
                coarse_b = path[j + 1].split(":")[0]
                key = f"{coarse_a}->{coarse_b}"
                transition_counts[key] = transition_counts.get(key, 0) + 1

        for step in step_logs:
            status = step["status"]
            if status not in ("PASS", "PLAN_DONE"):
                failure_type_counts[status] = failure_type_counts.get(status, 0) + 1

        for step in step_logs:
            status = step["status"]
            family = "PASS" if status == "PASS" else str(status).split(":")[0]
            if family not in ("PASS", "PLAN_DONE"):
                failure_family_counts[family] = failure_family_counts.get(family, 0) + 1

        run_analysis = {
            "run_id": run_id,
            "dataset": dataset_name,
            "method": method_name,
            "transition_counts": transition_counts,
            "failure_type_counts": failure_type_counts,
            "failure_family_counts": failure_family_counts,
        }

        if save_step_level:
            save_results_jsonl(step_logs, os.path.join(output_dir, "step_logs.jsonl"))

        if save_trajectory_level:
            save_results_jsonl(trajectory_logs, os.path.join(output_dir, "trajectory_logs.jsonl"))

        if save_problem_summary:
            save_result(problem_summary, os.path.join(output_dir, "summary.json"))

        if save_run_analysis:
            save_result(run_analysis, os.path.join(output_dir, "analysis.json"))

        if failure_examples:
            save_result(
                failure_examples,
                os.path.join(output_dir, "failure_examples.json"),
            )
            print(f"📝 failure_examples: {len(failure_examples)}개 유형 저장됨")

    finally:
        if log_fp is not None:
            log_fp.close()
            sys.stdout = stdout_backup
            print(f"sample debug output saved to: {debug_log_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.code_then_plan_repair <config.yaml>")
        sys.exit(1)

    run_code_then_plan_repair(sys.argv[1])