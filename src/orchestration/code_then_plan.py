"""
Code-Then-Plan

흐름:
Problem → G(Generate) → V(Evaluate)
         → [실패 시] Plan → Code → V
         → [계속 실패 시] Plan → Code → V
         → ... (budget 내 반복)

- 첫 시도는 generate 1회
- 실패하면 generic planner를 호출하고, 해당 plan으로 coder가 재생성
- 이후에도 실패하면 같은 generic planner -> plan_code cycle을 반복
- replanner / repair는 사용하지 않음
- max_calls는 전체 글로벌 budget으로 적용

call 단위:
- generate = 1 call
- plan = 1 call
- plan_code = 1 call
"""
from __future__ import annotations

import gc
import os
import sys
import time
from types import SimpleNamespace

import yaml

from src.models.hf_model_vllm import HFModel
from src.evaluation.metrics import summarize_failure_breakdown, summarize_phase1_results
from src.utils.io import save_result, save_results_jsonl, make_run_id
from src.utils.dataloader import load_task_and_adapter
from src.utils.prompt_loader import build_planner_prompt_for_sample, build_coder_prompt_for_sample
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


def _extract_entropy_fields(gen_result: dict) -> dict:
    """
    hf_model.generate() 반환값에서 entropy 관련 필드를 추출한다.
    hf_model.py가 수정되지 않은 환경에서도 안전하게 0.0으로 fallback한다.
    """
    return {
        "avg_entropy":          gen_result.get("avg_entropy", 0.0),
        "max_entropy":          gen_result.get("max_entropy", 0.0),
        "entropy_std":          gen_result.get("entropy_std", 0.0),
        "first_20pct_entropy":  gen_result.get("first_20pct_entropy", 0.0),
        "last_20pct_entropy":   gen_result.get("last_20pct_entropy", 0.0),
    }


def _extract_code_for_planner(adapter, sample, raw_text: str):
    if hasattr(adapter, "extract_code_for_planner"):
        return adapter.extract_code_for_planner(sample, raw_text)
    return adapter.extract_code(sample, raw_text)


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────

def run_code_then_plan(config_path: str):
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

    method_name = method_cfg.get("name", "code_then_plan")
    max_calls = budget_cfg.get("max_calls", 3)

    output_dir = output_cfg.get("dir", f"results/RUN/{dataset_name}/code_then_plan")

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
        print("🔄 Code-Then-Plan 실험")
        print("=" * 60)
        print(f"run_id              : {run_id}")
        print(f"dataset             : {dataset_name}")
        print(f"method              : {method_name}")
        print(f"model               : {model_name}")
        print(f"max_new_tokens      : {max_new_tokens}")
        print(f"planner_max_tokens  : {planner_max_tokens}")
        print(f"coder_max_tokens    : {coder_max_tokens}")
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
            backend=model_cfg.get("backend", "hf"),
            api_base=model_cfg.get("api_base", None),
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
            latest_code = None
            plan_attempt_count = 0

            # =========================================================
            # Stage 1: Direct generation
            # =========================================================
            if call_count < max_calls:
                initial_prompt = adapter.build_initial_prompt(sample)

                gen_start = time.perf_counter()
                try:
                    gen_result = model.generate(initial_prompt)
                except Exception as e:
                    gen_end = time.perf_counter()
                    print(f"  ⚠️ 모델 호출 실패 (토큰 초과 등), 스킵: {e}")
                    _overflow_record = _make_empty_output_record(
                        f"TokenOverflow: {str(e)[:300]}"
                    )
                    eval_results.append(SimpleNamespace(
                        status="TOKEN_OVERFLOW", passed=False,
                        exec_ok=False, test_pass=False,
                        tests_passed=0, tests_total=0,
                        error_type="TokenOverflow", error_stage="generate",
                        num_calls=0,
                    ))
                    trajectory_logs.append({
                        "run_id": run_id, "dataset": dataset_name,
                        "problem_id": problem_id, "method": method_name,
                        "trajectory_id": trajectory_id,
                        "num_steps": 0, "call_count": 0,
                        "final_status": "TOKEN_OVERFLOW", "failure_family": "TOKEN_OVERFLOW",
                        "final_tests_passed": 0, "final_tests_total": 0,
                        "total_input_tokens": 0, "total_output_tokens": 0,
                        "total_tokens": 0, "total_latency": gen_end - gen_start,
                        "num_exec_fail": 0, "num_test_fail": 0,
                        "transition_path": ["TOKEN_OVERFLOW"],
                        "budget_used": {"tokens": 0, "calls": 0, "latency": gen_end - gen_start},
                    })
                    gc.collect()
                    continue
                gen_end = time.perf_counter()
                latency_sec = gen_end - gen_start

                raw_text = gen_result["text"]
                input_tokens = gen_result["input_tokens"]
                output_tokens = gen_result["output_tokens"]
                total_tokens = gen_result["total_tokens"]
                #entropy_fields = _extract_entropy_fields(gen_result)

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
                        # ── entropy
                        # **entropy_fields,
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

                    if save_step_level:
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
                        # ── entropy
                        # **entropy_fields,
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

                    if save_step_level:
                        step_logs.append(step_entry)

                    pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
                    print(f"  generate: {pretty}  ")

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
            # Planning loop: generic plan -> plan_code 반복
            # =========================================================
            while (
                final_attempt_record is not None
                and not final_attempt_record.passed
                and call_count + 2 <= max_calls
            ):
                used_plan = True
                plan_attempt_count += 1

                # ── 1) plan
                planner_prompt = build_planner_prompt_for_sample(sample)

                orig_tokens = model.max_new_tokens
                model.max_new_tokens = planner_max_tokens
                try:
                    plan_start = time.perf_counter()
                    plan_gen_result = model.generate(planner_prompt)
                    plan_end = time.perf_counter()
                except Exception as e:
                    model.max_new_tokens = orig_tokens
                    print(f"  ⚠️ plan 모델 호출 실패 (토큰 초과 등), 이 문제 중단: {e}")
                    transition_path.append("TOKEN_OVERFLOW")
                    break
                finally:
                    model.max_new_tokens = orig_tokens

                plan_latency = plan_end - plan_start
                plan_raw_text = plan_gen_result["text"]
                plan_input_tokens = plan_gen_result["input_tokens"]
                plan_output_tokens = plan_gen_result["output_tokens"]
                plan_total_tokens = plan_gen_result["total_tokens"]
                # plan_entropy_fields = _extract_entropy_fields(plan_gen_result)

                call_count += 1
                cumulative_input_tokens += plan_input_tokens
                cumulative_output_tokens += plan_output_tokens
                cumulative_total_tokens += plan_total_tokens
                cumulative_latency += plan_latency

                planner_output = extract_planner_output(plan_raw_text)
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
                    # ── entropy (planner 자체의 불확실성)
                    # **plan_entropy_fields,
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
                    plan_step_entry["entry_point"] = sample.entry_point

                if save_step_level:
                    step_logs.append(plan_step_entry)
                print(f"  plan[{plan_attempt_count}]: 📝 "
                      f"{planner_output[:60].replace(chr(10), ' ')}...")

                # ── 2) plan_code
                coder_prompt = build_coder_prompt_for_sample(sample, planner_output)

                model.max_new_tokens = coder_max_tokens
                try:
                    code2_start = time.perf_counter()
                    code2_gen_result = model.generate(coder_prompt)
                    code2_end = time.perf_counter()
                except Exception as e:
                    model.max_new_tokens = orig_tokens
                    print(f"  ⚠️ plan_code 모델 호출 실패 (토큰 초과 등), 이 문제 중단: {e}")
                    transition_path.append("TOKEN_OVERFLOW")
                    break
                finally:
                    model.max_new_tokens = orig_tokens

                code2_latency = code2_end - code2_start
                code2_raw_text = code2_gen_result["text"]
                code2_input_tokens = code2_gen_result["input_tokens"]
                code2_output_tokens = code2_gen_result["output_tokens"]
                code2_total_tokens = code2_gen_result["total_tokens"]
                # code2_entropy_fields = _extract_entropy_fields(code2_gen_result)

                call_count += 1
                cumulative_input_tokens += code2_input_tokens
                cumulative_output_tokens += code2_output_tokens
                cumulative_total_tokens += code2_total_tokens
                cumulative_latency += code2_latency

                if code2_output_tokens == 0 or not code2_raw_text.strip():
                    final_attempt_record = _make_empty_output_record(
                        "Model output was empty (plan_code stage)."
                    )
                    final_exec_result = final_attempt_record
                    latest_code = None
                    current_status = final_attempt_record.status
                    transition_path.append(current_status)

                    code2_step_entry = {
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
                        "input_tokens": code2_input_tokens,
                        "output_tokens": code2_output_tokens,
                        "total_tokens": code2_total_tokens,
                        "latency_sec": code2_latency,
                        # ── entropy
                        # **code2_entropy_fields,
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
                        code2_step_entry["entry_point"] = sample.entry_point
                    if save_step_level:
                        step_logs.append(code2_step_entry)
                    print(f"  plan_code[{plan_attempt_count}]: ❌ {current_status}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=plan_attempt_count,
                        status=current_status,
                        prompt=coder_prompt,
                        raw_text=code2_raw_text,
                        generated_code=None,
                        error_type=final_attempt_record.error_type,
                        error_stage=final_attempt_record.error_stage,
                        error_message=final_attempt_record.error_message,
                    )

                else:
                    if hasattr(adapter, "extract_code_for_planner"):
                        code2_generated = adapter.extract_code_for_planner(sample, code2_raw_text)
                    else:
                        code2_generated = adapter.extract_code(sample, code2_raw_text)

                    if debug_mode == "sample":
                        _print_sample_execution_flow(
                            sample=sample,
                            prompt=coder_prompt,
                            raw_text=code2_raw_text,
                            generated_code=code2_generated,
                        )

                    exec_result = adapter.execute(sample, code2_generated)

                    if debug_mode == "sample":
                        _print_header("SAMPLE DEBUG :: EXEC RESULT OBJECT (plan_code)")
                        print(exec_result)
                        _print_header("SAMPLE DEBUG :: CLASSIFIED EXECUTION (plan_code)")
                        print(adapter.classify_execution(exec_result))

                    code2_record = adapter.make_attempt_record(
                        sample=sample,
                        method=method_name,
                        model_name=model_name,
                        attempt_idx=plan_attempt_count,
                        prompt=coder_prompt,
                        raw_output=code2_raw_text,
                        generated_code=code2_generated,
                        latency_sec=code2_latency,
                        exec_result=exec_result,
                    )

                    if debug_mode == "sample":
                        _print_header("SAMPLE DEBUG :: ATTEMPT RECORD (plan_code)")
                        print(code2_record)

                    final_attempt_record = code2_record
                    final_exec_result = exec_result
                    latest_code = code2_generated
                    current_status = code2_record.status
                    transition_path.append(current_status)

                    if str(current_status).startswith("EXEC_FAIL"):
                        num_exec_fail += 1
                    if str(current_status).startswith("TEST_FAIL"):
                        num_test_fail += 1

                    code2_step_entry = {
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
                        "input_tokens": code2_input_tokens,
                        "output_tokens": code2_output_tokens,
                        "total_tokens": code2_total_tokens,
                        "latency_sec": code2_latency,
                        # ── entropy
                        # **code2_entropy_fields,
                        "code": code2_generated if save_code else None,
                        "planner_output": None,
                        "exec_ok": code2_record.exec_ok,
                        "test_pass": code2_record.test_pass,
                        "status": current_status,
                        "error_type": code2_record.error_type,
                        "error_stage": code2_record.error_stage,
                        "error_message": code2_record.error_message,
                        "tests_passed": code2_record.tests_passed,
                        "tests_total": code2_record.tests_total,
                        "code_length": len(code2_generated) if code2_generated else 0,
                        "selected": None,
                        "selection_rank": None,
                    }
                    if hasattr(sample, "entry_point"):
                        code2_step_entry["entry_point"] = sample.entry_point

                    if save_step_level:
                        step_logs.append(code2_step_entry)

                    pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
                    print(f"  plan_code[{plan_attempt_count}]: {pretty}  ")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=plan_attempt_count,
                        status=current_status,
                        prompt=coder_prompt,
                        raw_text=code2_raw_text,
                        generated_code=code2_generated,
                        error_type=code2_record.error_type,
                        error_stage=code2_record.error_stage,
                        error_message=code2_record.error_message,
                    )

                if final_attempt_record.passed:
                    break

            if final_exec_result is None:
                final_exec_result = _make_empty_output_record("No executable result was produced.")
            if final_attempt_record is None:
                final_attempt_record = _make_empty_output_record("No attempt record was produced.")
            
            setattr(final_exec_result, "num_calls", call_count)
            eval_results.append(SimpleNamespace(
                status=final_attempt_record.status,
                passed=final_attempt_record.passed,
                exec_ok=final_attempt_record.exec_ok,
                test_pass=final_attempt_record.test_pass,
                tests_passed=final_attempt_record.tests_passed,
                tests_total=final_attempt_record.tests_total,
                error_type=getattr(final_attempt_record, "error_type", None),
                error_stage=getattr(final_attempt_record, "error_stage", None),
                num_calls=call_count,
            ))
            
            final_status = final_attempt_record.status
            failure_family = "PASS" if final_status == "PASS" else str(final_status).split(":")[0]

            # trajectory 수준 entropy 시계열 (plan_code step만 — 코드 생성 step 기준)
            code_steps = [
                s for s in step_logs
                if s["trajectory_id"] == trajectory_id
                and s["stage"] in ("generate", "plan_code")
            ]
            # entropy_series = [s["avg_entropy"] for s in code_steps]

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
                # ── entropy 시계열 요약
                # "entropy_series": entropy_series,
                # "initial_avg_entropy": entropy_series[0] if entropy_series else None,
                "budget_used": {
                    "tokens": cumulative_total_tokens,
                    "calls": call_count,
                    "latency": cumulative_latency,
                },
                "recovered_by": (
                    "generate" if (not used_plan and final_status == "PASS")
                    else "plan_code" if (used_plan and final_status == "PASS")
                    else None
                ),
                "plan_recovery_attempt": (
                    plan_attempt_count if (used_plan and final_status == "PASS")
                    else None
                ),
            }

            if save_trajectory_level:
                trajectory_logs.append(trajectory_entry)

            del final_exec_result
            gc.collect()

        summary = summarize_phase1_results(eval_results, k=max_calls)
        success_key = f"success@{max_calls}"
        
        print(f"\n{'=' * 60}")
        print("📊 결과 요약")
        print(f"  총 문제: {summary['total']}")
        print(f"  성공: {summary['success']}")
        print(f"  실행 성공: {summary['exec_success']}")
        print(f"  {success_key}: {summary[success_key]:.4f}")
        print(f"  execution_success_rate: {summary['execution_success_rate']:.4f}")
        print(f"  conditional_success: {summary['conditional_success']:.4f}")
        print(f"  AUSC: {summary['ausc']:.4f}")
        print(f"{'=' * 60}")

        extra_summary = summarize_failure_breakdown(eval_results)
        print(f"  code_failed: {extra_summary['code_failed']}")
        print(f"  define_test_failed: {extra_summary['define_test_failed']}")
        print(f"  run_test_failed: {extra_summary['run_test_failed']}")
        print(f"{'=' * 60}")

        n_used_plan = sum(1 for t in trajectory_logs if t.get("used_plan", False))
        n_plan_recovered = sum(
            1 for t in trajectory_logs
            if t.get("recovered_by") == "plan_code"
        )

        print(
            f"  plan 사용: {n_used_plan}/{len(trajectory_logs)} "
            f"({n_used_plan/len(trajectory_logs)*100:.1f}%)"
            if len(trajectory_logs) > 0 else "  plan 사용: 0/0"
        )
        print(
            f"  plan 복구 성공: {n_plan_recovered}/{n_used_plan} "
            f"({n_plan_recovered/n_used_plan*100:.1f}%)"
            if n_used_plan > 0 else "  plan 복구 성공: 0/0"
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
            "max_calls": max_calls,
            "total_problems": summary["total"],
            "num_success": summary["success"],
            "success_metric_name": success_key,
            "success_at_k": summary[success_key],
            "success_at_k_curve": summary["success_at_k_curve"],
            "ausc": summary["ausc"],
            "execution_success_rate": summary["execution_success_rate"],
            "conditional_success": summary["conditional_success"],
            "avg_tokens": avg_tokens,
            "avg_latency": avg_latency,
            "avg_calls": avg_calls,
            "extra_summary": extra_summary,
            "plan_stats": {
                "used_plan": n_used_plan,
                "plan_recovered": n_plan_recovered,
                "plan_recovery_rate": n_plan_recovered / n_used_plan if n_used_plan > 0 else 0.0,
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
        print("Usage: python -m src.orchestration.code_then_plan <config.yaml>")
        sys.exit(1)

    run_code_then_plan(sys.argv[1])