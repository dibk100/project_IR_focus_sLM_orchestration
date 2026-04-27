"""
Policy

stage별 오류 유형과, 실패 반복 패턴을 기반으로, repair / replan 중 적절한 전략을 동적으로 선택하는 policy를 구현하고자 함.
제한된 budget 내에서 문제 해결 확률을 극대화하는 것을 목표로 하고 싶었으나, 사실상 간단한 rule-based로 구현됨.


State = {
    failure_type : 현재 실행 결과의 상위 실패 단계 (PASS / EXEC_FAIL / TEST_FAIL)
    error_type : 실패의 구체적인 에러 타입 
                 (AssertionError / SyntaxError / TypeError / ...)
    previous_state : 직전 step의 실패 상태 
                     (prev_failure_type, prev_error_type)
    repetition_pattern : 동일한 (failure_type, error_type)가 
                         연속으로 반복된 횟수
    last_action : 직전에 수행한 전략 
                  (generate / repair / plan)
    plan_budget_usage : 현재까지 사용한 planning 횟수 
                        (num_plan_calls)
}

Action = {
    generate : 문제 입력만을 기반으로 초기 코드를 생성하는 단계

    repair : 이전에 생성한 코드와 실패 피드백(error message)을 바탕으로 
             코드를 수정하는 단계

    re-plan : 문제 입력(및 필요 시 이전 실패 정보)을 바탕으로 
           해결 계획을 생성한 뒤, 그 계획에 따라 코드를 다시 생성하는 단계
}
Rules:

1. Initial Rule
   - 처음에는 항상 generate를 수행한다.
   - 첫 번째 plan 호출은 문제 입력만을 기반으로 plan을 생성한다.
   - 두 번째 이후의 plan 호출은 re-plan으로 간주하며,
     이전 plan, 실패 코드, 실패 상태, error_type, error_message를 함께 사용한다.

2. AssertionError Rule
   if failure_type == TEST_FAIL and error_type == AssertionError:
       → plan / re-plan
   - 첫 generate 이후 AssertionError가 발생하면 plan으로 전환한다.
   - plan 이후에도 같은 AssertionError가 발생하면 repair가 아니라 re-plan으로 전환한다.
   - AssertionError는 semantic failure로 간주하므로 repair의 기본 대상에서 제외한다.

3. Restricted Repair Rule
   if failure_type == EXEC_FAIL and error_type in {
       SyntaxError, NameError, TypeError, ImportError
   }:
       → repair
   - repair는 execution-level 오류를 수정하기 위한 보조 전략으로만 사용한다.
   - repair는 문제 단위로 최대 K번만 허용한다.
   - 권장 설정은 K = 1이다.
   - TEST_FAIL, 특히 AssertionError에는 repair를 반복 적용하지 않는다.

4. Repair-to-Replan Rule
   if repair 이후에도 PASS가 되지 않으면:
       → re-plan
   - repair를 여러 번 반복하지 않는다.
   - repair 이후 동일하거나 새로운 실패가 발생하면, 현재 코드 패치가 한계에 도달한 것으로 보고 re-plan으로 전환한다.

5. Stagnation Rule
   if (failure_type, error_type) repeated ≥ K:
       → re-plan
   - 같은 실패 패턴이 K번 이상 반복되면 동일 전략 반복을 중단한다.
   - 기존 코드 수정보다 문제 재해석이 필요하다고 판단하여 re-plan으로 전환한다.
   - 권장 설정은 K = 2이다.

6. Default Planning Rule
   그 외의 경우:
       → plan / re-plan
   - 기존의 Repair Default Rule을 제거한다.
   - repair를 기본 전략으로 사용하지 않는다.
   - 명확한 execution-level 오류가 아닌 경우에는 plan 또는 re-plan을 기본 선택으로 둔다.

7. Budget Constraint Rule
   if plan_budget_usage ≥ max_plan_calls:
       → plan / re-plan 불가 → 종료
   - plan과 re-plan은 동일한 planning budget을 공유한다.
   - 예산이 소진된 경우 더 이상 planning 계열 전략을 호출하지 않고 종료한다.

8. Global Budget Rule
   if total_calls ≥ max_calls:
       → 종료
   - 전체 호출 횟수가 max_calls에 도달하면 더 이상 시도하지 않고 종료한다.
"""
import gc
import os
import sys
import time
import yaml
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.models.hf_model import HFModel
from src.evaluation.metrics import summarize_failure_breakdown, summarize_phase1_results
from src.utils.io import save_result, save_results_jsonl, make_run_id
from src.utils.dataloader import load_task_and_adapter
from src.utils.prompt_loader import (
    build_planner_prompt_for_sample,
    build_coder_prompt_for_sample,
    build_repair_prompt_for_sample,
    build_replanner_prompt_for_sample,
)
from src.utils.prompting.planner_coder import extract_planner_output


# ============================================================
# Debug helpers
# ============================================================

def _shorten(text, max_len: int = 10000) -> str:
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

    _print_header("SAMPLE DEBUG :: STEP 2. RAW MODEL OUTPUT")
    print(_shorten(raw_text))

    _print_header("SAMPLE DEBUG :: STEP 3. EXECUTABLE CODE")
    if generated_code is None:
        print("(empty or not yet extracted)")
        return
    print(_shorten(generated_code))


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


def _make_empty_output_attempt_record(status_message: str = "empty_output"):
    return SimpleNamespace(
        status="CODE_FAIL:empty_output",
        tests_passed=0,
        tests_total=None,
        passed=False,
        exec_ok=False,
        test_pass=False,
        error_type="empty_output",
        error_stage="code",
        error_message=status_message,
    )


def _maybe_extract_code_for_repair(adapter, sample, raw_text: str):
    if hasattr(adapter, "extract_code_for_repair"):
        return adapter.extract_code_for_repair(sample, raw_text)
    if hasattr(adapter, "extract_code_for_planner"):
        return adapter.extract_code_for_planner(sample, raw_text)
    return adapter.extract_code(sample, raw_text)


def _extract_code_for_plan(adapter, sample, raw_text: str):
    if hasattr(adapter, "extract_code_for_planner"):
        return adapter.extract_code_for_planner(sample, raw_text)
    return adapter.extract_code(sample, raw_text)


# ============================================================
# Token stats helpers
# ============================================================

def _compute_token_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "avg": 0.0, "count": 0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "count": len(values),
    }


# ============================================================
# Policy helpers
# ============================================================

@dataclass
class StepRecord:
    fail: str
    error_type: str
    action: str   # generate / repair / plan / replan


def coarse_fail(status: str) -> str:
    if status == "PASS":
        return "PASS"
    return str(status).split(":")[0]


def normalize_error_type(error_type: Optional[str]) -> str:
    if error_type is None:
        return "NONE"
    s = str(error_type).strip()
    return s if s else "NONE"


def make_signature(fail: str, error_type: str) -> Tuple[str, str]:
    return (fail, error_type)


def num_plan_calls(history: List[StepRecord]) -> int:
    return sum(1 for x in history if x.action in {"plan", "replan"})


def same_signature_run_length(history: List[StepRecord]) -> int:
    if not history:
        return 0
    last_sig = make_signature(history[-1].fail, history[-1].error_type)
    cnt = 1
    for step in reversed(history[:-1]):
        if make_signature(step.fail, step.error_type) == last_sig:
            cnt += 1
        else:
            break
    return cnt


def same_fail_family_run_length(history: List[StepRecord]) -> int:
    if not history:
        return 0
    last_fail = history[-1].fail
    cnt = 1
    for step in reversed(history[:-1]):
        if step.fail == last_fail:
            cnt += 1
        else:
            break
    return cnt


def make_policy_state(history: List[StepRecord]) -> Dict[str, Any]:
    if not history:
        return {
            "prev_fail": "NONE",
            "prev_error": "NONE",
            "last_fail": "NONE",
            "last_error": "NONE",
            "same_signature_run_length": 0,
            "same_fail_family_run_length": 0,
            "last_action": "NONE",
            "num_plan_calls": 0,
        }

    last = history[-1]
    prev = history[-2] if len(history) >= 2 else None

    return {
        "prev_fail": prev.fail if prev else "NONE",
        "prev_error": prev.error_type if prev else "NONE",
        "last_fail": last.fail,
        "last_error": last.error_type,
        "same_signature_run_length": same_signature_run_length(history),
        "same_fail_family_run_length": same_fail_family_run_length(history),
        "last_action": last.action,
        "num_plan_calls": num_plan_calls(history),
    }


def plan_available(history: List[StepRecord], policy_cfg: Dict[str, Any]) -> bool:
    max_plan_calls = policy_cfg.get("max_plan_calls", 3)
    return num_plan_calls(history) < max_plan_calls


def repair_available(total_repair_count: int, policy_cfg: Dict[str, Any]) -> bool:
    max_repair_steps = policy_cfg.get("max_repair_steps", 1)
    return total_repair_count < max_repair_steps


def choose_next_action(
    *,
    history: List[StepRecord],
    current_fail: str,
    current_error: str,
    policy_cfg: Dict[str, Any],
    total_repair_count: int,
) -> str:
    """Route failures to repair or plan/replan under a limited budget."""
    if current_fail == "PASS":
        return "stop"

    stagnation_threshold = policy_cfg.get("stagnation_threshold", 2)

    # Semantic failure: re-interpret the task instead of repeatedly patching code.
    if current_fail == "TEST_FAIL" and current_error == "AssertionError":
        return "plan"

    # Repeated same failure means the current strategy is stagnant.
    if same_signature_run_length(history) >= stagnation_threshold:
        return "plan"

    # Empty/extraction/code failures are usually better handled by regenerate-with-plan.
    if current_fail == "CODE_FAIL":
        return "plan"

    # Execution failures get at most a small number of cheap repair attempts.
    repairable_exec_errors = {
        "SyntaxError",
        "NameError",
        "TypeError",
        "ImportError",
        "IndentationError",
        "UnboundLocalError",
    }
    if current_fail == "EXEC_FAIL":
        if current_error in repairable_exec_errors and repair_available(total_repair_count, policy_cfg):
            return "repair"
        return "plan"

    # Default: prefer plan/replan over low-yield repeated repair.
    return "plan"


# ============================================================
# Step runners
# ============================================================

def _run_generate_step(
    *,
    model,
    adapter,
    sample,
    method_name: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    debug_mode: str,
    attempt_idx: int,
):
    orig_tokens = model.max_new_tokens
    model.max_new_tokens = max_new_tokens
    try:
        start = time.perf_counter()
        gen_result = model.generate(prompt)
        end = time.perf_counter()
    finally:
        model.max_new_tokens = orig_tokens

    latency_sec = end - start
    raw_text = gen_result["text"]
    input_tokens = gen_result["input_tokens"]
    output_tokens = gen_result["output_tokens"]
    total_tokens = gen_result["total_tokens"]

    if debug_mode == "sample":
        _print_sample_execution_flow(sample=sample, prompt=prompt, raw_text=raw_text)

    if output_tokens == 0 or not raw_text.strip():
        attempt_record = _make_empty_output_attempt_record("empty_output: no code was generated.")
        generated_code = None
        exec_result = attempt_record
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
            latency_sec=latency_sec,
            exec_result=exec_result,
        )

    return {
        "prompt": prompt,
        "raw_text": raw_text,
        "generated_code": generated_code,
        "exec_result": exec_result,
        "attempt_record": attempt_record,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_sec": latency_sec,
    }


def _run_plan_then_code_step(
    *,
    model,
    adapter,
    sample,
    method_name: str,
    model_name: str,
    planner_max_tokens: int,
    coder_max_tokens: int,
    debug_mode: str,
    attempt_idx: int,
    previous_plan: str | None = None,
    failing_status: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    previous_code: str | None = None,
):
    if previous_plan is None:
        planner_prompt = build_planner_prompt_for_sample(sample)
    else:
        planner_prompt = build_replanner_prompt_for_sample(
            sample=sample,
            previous_plan=previous_plan,
            failing_status=failing_status,
            error_type=error_type,
            error_message=error_message,
            previous_code=previous_code,
        )

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

    planner_output = extract_planner_output(plan_raw_text)
    coder_prompt = build_coder_prompt_for_sample(sample, planner_output)

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

    if debug_mode == "sample":
        _print_sample_execution_flow(sample=sample, prompt=coder_prompt, raw_text=code_raw_text)

    if code_output_tokens == 0 or not code_raw_text.strip():
        attempt_record = _make_empty_output_attempt_record("Model output was empty (plan_code stage).")
        generated_code = None
        exec_result = attempt_record
    else:
        generated_code = _extract_code_for_plan(adapter, sample, code_raw_text)
        exec_result = adapter.execute(sample, generated_code)

        attempt_record = adapter.make_attempt_record(
            sample=sample,
            method=method_name,
            model_name=model_name,
            attempt_idx=attempt_idx,
            prompt=coder_prompt,
            raw_output=code_raw_text,
            generated_code=generated_code,
            latency_sec=code_latency,
            exec_result=exec_result,
        )

    return {
        "planner_prompt": planner_prompt,
        "planner_output": planner_output,
        "plan_raw_text": plan_raw_text,
        "plan_input_tokens": plan_input_tokens,
        "plan_output_tokens": plan_output_tokens,
        "plan_total_tokens": plan_total_tokens,
        "plan_latency": plan_latency,
        "coder_prompt": coder_prompt,
        "raw_text": code_raw_text,
        "generated_code": generated_code,
        "exec_result": exec_result,
        "attempt_record": attempt_record,
        "input_tokens": code_input_tokens,
        "output_tokens": code_output_tokens,
        "total_tokens": code_total_tokens,
        "latency_sec": code_latency,
    }


def _run_repair_step(
    *,
    model,
    adapter,
    sample,
    method_name: str,
    model_name: str,
    repair_max_tokens: int,
    previous_code: str | None,
    error_message: str | None,
    failing_status: str,
    planner_output: str | None,
    debug_mode: str,
    attempt_idx: int,
):
    repair_prompt = build_repair_prompt_for_sample(
        sample=sample,
        previous_code=previous_code,
        error_message=error_message,
        failing_status=failing_status,
        planner_output=planner_output,
    )

    orig_tokens = model.max_new_tokens
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

    if debug_mode == "sample":
        _print_sample_execution_flow(sample=sample, prompt=repair_prompt, raw_text=repair_raw_text)

    if repair_output_tokens == 0 or not repair_raw_text.strip():
        attempt_record = _make_empty_output_attempt_record("Model output was empty (repair stage).")
        repaired_code = None
        exec_result = attempt_record
    else:
        repaired_code = _maybe_extract_code_for_repair(adapter, sample, repair_raw_text)
        exec_result = adapter.execute(sample, repaired_code)

        attempt_record = adapter.make_attempt_record(
            sample=sample,
            method=method_name,
            model_name=model_name,
            attempt_idx=attempt_idx,
            prompt=repair_prompt,
            raw_output=repair_raw_text,
            generated_code=repaired_code,
            latency_sec=repair_latency,
            exec_result=exec_result,
        )

    return {
        "prompt": repair_prompt,
        "raw_text": repair_raw_text,
        "generated_code": repaired_code,
        "exec_result": exec_result,
        "attempt_record": attempt_record,
        "input_tokens": repair_input_tokens,
        "output_tokens": repair_output_tokens,
        "total_tokens": repair_total_tokens,
        "latency_sec": repair_latency,
    }


# ============================================================
# Logging helpers
# ============================================================

def _make_policy_state_for_log(history: List[StepRecord]) -> Dict[str, Any]:
    return make_policy_state(history)


def _append_step_log(
    step_logs: List[Dict[str, Any]],
    *,
    run_id: str,
    dataset_name: str,
    problem_id: str,
    method_name: str,
    trajectory_id: str,
    step_id: int,
    call_index: int,
    stage: str,
    policy_action: str,
    policy_state: Optional[Dict[str, Any]],
    policy_stop: bool,
    policy_stop_reason: Optional[str],
    sample,
    save_code: bool,
    generated_code: Optional[str],
    attempt_record,
    input_tokens,
    output_tokens,
    total_tokens,
    latency_sec,
    planner_output: Optional[str] = None,
    is_retry: bool = False,
    is_repair: bool = False,
    is_planner: bool = False,
):
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
        "policy_action": policy_action,
        "policy_state": policy_state,
        "policy_stop": policy_stop,
        "policy_stop_reason": policy_stop_reason,
        "is_retry": is_retry,
        "is_repair": is_repair,
        "is_planner": is_planner,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_sec": latency_sec,
        "code": generated_code if save_code else None,
        "planner_output": planner_output if save_code else None,
        "exec_ok": getattr(attempt_record, "exec_ok", None),
        "test_pass": getattr(attempt_record, "test_pass", None),
        "status": getattr(attempt_record, "status", stage),
        "error_type": getattr(attempt_record, "error_type", None),
        "error_stage": getattr(attempt_record, "error_stage", None),
        "error_message": getattr(attempt_record, "error_message", None),
        "tests_passed": getattr(attempt_record, "tests_passed", None),
        "tests_total": getattr(attempt_record, "tests_total", None),
        "code_length": len(generated_code) if generated_code else 0,
        "selected": None,
        "selection_rank": None,
    }
    if hasattr(sample, "entry_point"):
        entry["entry_point"] = sample.entry_point
    step_logs.append(entry)


# ============================================================
# Main runner
# ============================================================

def run_policy_loop(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg = config.get("run", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    method_cfg = config.get("method", {})
    budget_cfg = config.get("budget", {})
    policy_cfg = config.get("policy", {})
    output_cfg = config.get("output", {})
    logging_cfg = config.get("logging", {})
    debug_cfg = config.get("debug", {})

    run_id = make_run_id(config)
    seed = run_cfg.get("seed", 42)

    stagnation_threshold = policy_cfg.get("stagnation_threshold", 2)

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

    method_name = method_cfg.get("name", "policy_loop")
    max_calls = budget_cfg.get("max_calls", 10)

    max_plan_calls = policy_cfg.get("max_plan_calls", 3)
    max_repair_steps = policy_cfg.get("max_repair_steps", 3)

    output_dir = output_cfg.get("dir", f"results/RUN/{dataset_name}/policy_loop")

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)
    save_failure_examples = logging_cfg.get("failure_examples", True)

    debug_mode = debug_cfg.get("mode", "run")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🔄 Policy Loop 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"model               : {model_name}")
    print(f"max_new_tokens      : {max_new_tokens}")
    print(f"planner_max_tokens  : {planner_max_tokens}")
    print(f"repair_max_tokens   : {repair_max_tokens}")
    print(f"max_calls           : {max_calls}")
    print(f"max_plan_calls      : {max_plan_calls}")
    print(f"max_repair_steps    : {max_repair_steps}")
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
            "policy": policy_cfg,
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

    step_logs: List[Dict[str, Any]] = []
    trajectory_logs: List[Dict[str, Any]] = []
    eval_results = []
    failure_examples = {}

    # run-level token tracking
    all_input_tokens: List[int] = []
    all_output_tokens: List[int] = []

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

        # problem-level token tracking
        problem_input_tokens: List[int] = []
        problem_output_tokens: List[int] = []

        call_count = 0
        total_repair_count = 0
        num_exec_fail = 0
        num_test_fail = 0
        transition_path: List[str] = []

        final_attempt_record = None
        final_exec_result = None
        recovered_by = None
        stop_reason = None

        previous_code = None
        planner_output = None
        history: List[StepRecord] = []

        step_id = 0
        attempt_idx = 0

        initial_prompt = adapter.build_initial_prompt(sample)

        gen = _run_generate_step(
            model=model,
            adapter=adapter,
            sample=sample,
            method_name=method_name,
            model_name=model_name,
            prompt=initial_prompt,
            max_new_tokens=max_new_tokens,
            debug_mode=debug_mode,
            attempt_idx=attempt_idx,
        )

        # token tracking
        all_input_tokens.append(gen["input_tokens"])
        all_output_tokens.append(gen["output_tokens"])
        problem_input_tokens.append(gen["input_tokens"])
        problem_output_tokens.append(gen["output_tokens"])

        call_count += 1
        cumulative_input_tokens += gen["input_tokens"]
        cumulative_output_tokens += gen["output_tokens"]
        cumulative_total_tokens += gen["total_tokens"]
        cumulative_latency += gen["latency_sec"]

        final_attempt_record = gen["attempt_record"]
        final_exec_result = gen["exec_result"]
        previous_code = gen["generated_code"]

        current_status = final_attempt_record.status
        current_fail = coarse_fail(current_status)
        current_error = normalize_error_type(final_attempt_record.error_type)
        transition_path.append(f"{current_fail}:{current_error}")

        if current_fail == "EXEC_FAIL":
            num_exec_fail += 1
        if current_fail == "TEST_FAIL":
            num_test_fail += 1

        history.append(StepRecord(fail=current_fail, error_type=current_error, action="generate"))

        _append_step_log(
            step_logs,
            run_id=run_id,
            dataset_name=dataset_name,
            problem_id=problem_id,
            method_name=method_name,
            trajectory_id=trajectory_id,
            step_id=step_id,
            call_index=call_count - 1,
            stage="generate",
            policy_action="generate",
            policy_state=None,
            policy_stop=False,
            policy_stop_reason=None,
            sample=sample,
            save_code=save_code,
            generated_code=gen["generated_code"],
            attempt_record=final_attempt_record,
            input_tokens=gen["input_tokens"],
            output_tokens=gen["output_tokens"],
            total_tokens=gen["total_tokens"],
            latency_sec=gen["latency_sec"],
        )

        pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
        print(f"  generate: {pretty}")

        _collect_failure_example(
            failure_examples,
            problem_id=problem_id,
            attempt_idx=attempt_idx,
            status=current_status,
            prompt=initial_prompt,
            raw_text=gen["raw_text"],
            generated_code=gen["generated_code"],
            error_type=final_attempt_record.error_type,
            error_stage=final_attempt_record.error_stage,
            error_message=final_attempt_record.error_message,
        )

        attempt_idx += 1
        step_id += 1

        if final_attempt_record.passed:
            recovered_by = "generate"
        else:
            forced_next_action = None

            while not final_attempt_record.passed:
                if call_count >= max_calls:
                    stop_reason = "max_calls_reached"
                    print(f"  stop: 🛑 {stop_reason}")
                    break

                next_action = forced_next_action or choose_next_action(
                    history=history,
                    current_fail=current_fail,
                    current_error=current_error,
                    policy_cfg=policy_cfg,
                    total_repair_count=total_repair_count,
                )
                forced_next_action = None

                if next_action == "stop":
                    stop_reason = "policy_stop"
                    print(f"  stop: 🛑 {stop_reason}")
                    break

                if next_action == "plan":
                    if not plan_available(history, policy_cfg):
                        stop_reason = "plan_budget_exhausted"
                        print(f"  stop: 🛑 {stop_reason}")
                        break

                    plan_action_name = "plan" if planner_output is None else "replan"

                    out = _run_plan_then_code_step(
                        model=model,
                        adapter=adapter,
                        sample=sample,
                        method_name=method_name,
                        model_name=model_name,
                        planner_max_tokens=planner_max_tokens,
                        coder_max_tokens=coder_max_tokens,
                        debug_mode=debug_mode,
                        attempt_idx=attempt_idx,
                        previous_plan=planner_output,
                        failing_status=final_attempt_record.status,
                        error_type=final_attempt_record.error_type,
                        error_message=final_attempt_record.error_message,
                        previous_code=previous_code,
                    )

                    # token tracking: planner call
                    all_input_tokens.append(out["plan_input_tokens"])
                    all_output_tokens.append(out["plan_output_tokens"])
                    problem_input_tokens.append(out["plan_input_tokens"])
                    problem_output_tokens.append(out["plan_output_tokens"])

                    call_count += 1
                    cumulative_input_tokens += out["plan_input_tokens"]
                    cumulative_output_tokens += out["plan_output_tokens"]
                    cumulative_total_tokens += out["plan_total_tokens"]
                    cumulative_latency += out["plan_latency"]

                    _append_step_log(
                        step_logs,
                        run_id=run_id,
                        dataset_name=dataset_name,
                        problem_id=problem_id,
                        method_name=method_name,
                        trajectory_id=trajectory_id,
                        step_id=step_id,
                        call_index=call_count - 1,
                        stage=plan_action_name,
                        policy_action=plan_action_name,
                        policy_state=_make_policy_state_for_log(history),
                        policy_stop=False,
                        policy_stop_reason=None,
                        sample=sample,
                        save_code=save_code,
                        generated_code=None,
                        attempt_record=SimpleNamespace(
                            status="PLAN_DONE",
                            exec_ok=None,
                            test_pass=None,
                            error_type=None,
                            error_stage=None,
                            error_message=None,
                            tests_passed=None,
                            tests_total=None,
                        ),
                        input_tokens=out["plan_input_tokens"],
                        output_tokens=out["plan_output_tokens"],
                        total_tokens=out["plan_total_tokens"],
                        latency_sec=out["plan_latency"],
                        planner_output=out["planner_output"],
                        is_planner=True,
                    )
                    transition_path.append(f"{plan_action_name.upper()}_DONE")
                    step_id += 1

                    # token tracking: plan_code call
                    all_input_tokens.append(out["input_tokens"])
                    all_output_tokens.append(out["output_tokens"])
                    problem_input_tokens.append(out["input_tokens"])
                    problem_output_tokens.append(out["output_tokens"])

                    call_count += 1
                    cumulative_input_tokens += out["input_tokens"]
                    cumulative_output_tokens += out["output_tokens"]
                    cumulative_total_tokens += out["total_tokens"]
                    cumulative_latency += out["latency_sec"]

                    final_attempt_record = out["attempt_record"]
                    final_exec_result = out["exec_result"]
                    previous_code = out["generated_code"]
                    planner_output = out["planner_output"]

                    current_status = final_attempt_record.status
                    current_fail = coarse_fail(current_status)
                    current_error = normalize_error_type(final_attempt_record.error_type)
                    transition_path.append(f"{current_fail}:{current_error}")

                    if current_fail == "EXEC_FAIL":
                        num_exec_fail += 1
                    if current_fail == "TEST_FAIL":
                        num_test_fail += 1

                    history.append(StepRecord(fail=current_fail, error_type=current_error, action=plan_action_name))

                    _append_step_log(
                        step_logs,
                        run_id=run_id,
                        dataset_name=dataset_name,
                        problem_id=problem_id,
                        method_name=method_name,
                        trajectory_id=trajectory_id,
                        step_id=step_id,
                        call_index=call_count - 1,
                        stage="plan_code",
                        policy_action=plan_action_name,
                        policy_state=_make_policy_state_for_log(history[:-1]),
                        policy_stop=False,
                        policy_stop_reason=None,
                        sample=sample,
                        save_code=save_code,
                        generated_code=out["generated_code"],
                        attempt_record=final_attempt_record,
                        input_tokens=out["input_tokens"],
                        output_tokens=out["output_tokens"],
                        total_tokens=out["total_tokens"],
                        latency_sec=out["latency_sec"],
                    )
                    step_id += 1

                    pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
                    print(f"  plan_code: {pretty}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=attempt_idx,
                        status=current_status,
                        prompt=out["coder_prompt"],
                        raw_text=out["raw_text"],
                        generated_code=out["generated_code"],
                        error_type=final_attempt_record.error_type,
                        error_stage=final_attempt_record.error_stage,
                        error_message=final_attempt_record.error_message,
                    )
                    attempt_idx += 1

                    if final_attempt_record.passed:
                        recovered_by = "plan_code" if plan_action_name == "plan" else "replan_code"
                        break

                    if current_fail == "TEST_FAIL" and current_error == "AssertionError":
                        forced_next_action = "plan"
                        continue

                repair_count = 0
                while repair_available(total_repair_count, policy_cfg) and not final_attempt_record.passed:
                    if call_count >= max_calls:
                        stop_reason = "max_calls_reached"
                        print(f"  stop: 🛑 {stop_reason}")
                        break

                    out = _run_repair_step(
                        model=model,
                        adapter=adapter,
                        sample=sample,
                        method_name=method_name,
                        model_name=model_name,
                        repair_max_tokens=repair_max_tokens,
                        previous_code=previous_code,
                        error_message=final_attempt_record.error_message,
                        failing_status=final_attempt_record.status,
                        planner_output=planner_output,
                        debug_mode=debug_mode,
                        attempt_idx=attempt_idx,
                    )

                    # token tracking: repair call
                    all_input_tokens.append(out["input_tokens"])
                    all_output_tokens.append(out["output_tokens"])
                    problem_input_tokens.append(out["input_tokens"])
                    problem_output_tokens.append(out["output_tokens"])

                    call_count += 1
                    cumulative_input_tokens += out["input_tokens"]
                    cumulative_output_tokens += out["output_tokens"]
                    cumulative_total_tokens += out["total_tokens"]
                    cumulative_latency += out["latency_sec"]

                    final_attempt_record = out["attempt_record"]
                    final_exec_result = out["exec_result"]
                    previous_code = out["generated_code"]

                    current_status = final_attempt_record.status
                    current_fail = coarse_fail(current_status)
                    current_error = normalize_error_type(final_attempt_record.error_type)
                    transition_path.append(f"{current_fail}:{current_error}")

                    if current_fail == "EXEC_FAIL":
                        num_exec_fail += 1
                    if current_fail == "TEST_FAIL":
                        num_test_fail += 1

                    history.append(StepRecord(fail=current_fail, error_type=current_error, action="repair"))

                    _append_step_log(
                        step_logs,
                        run_id=run_id,
                        dataset_name=dataset_name,
                        problem_id=problem_id,
                        method_name=method_name,
                        trajectory_id=trajectory_id,
                        step_id=step_id,
                        call_index=call_count - 1,
                        stage="repair",
                        policy_action="repair",
                        policy_state=_make_policy_state_for_log(history[:-1]),
                        policy_stop=False,
                        policy_stop_reason=None,
                        sample=sample,
                        save_code=save_code,
                        generated_code=out["generated_code"],
                        attempt_record=final_attempt_record,
                        input_tokens=out["input_tokens"],
                        output_tokens=out["output_tokens"],
                        total_tokens=out["total_tokens"],
                        latency_sec=out["latency_sec"],
                        is_repair=True,
                        is_retry=total_repair_count > 0,
                    )
                    step_id += 1

                    pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
                    print(f"  repair: {pretty}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=attempt_idx,
                        status=current_status,
                        prompt=out["prompt"],
                        raw_text=out["raw_text"],
                        generated_code=out["generated_code"],
                        error_type=final_attempt_record.error_type,
                        error_stage=final_attempt_record.error_stage,
                        error_message=final_attempt_record.error_message,
                    )
                    attempt_idx += 1
                    repair_count += 1
                    total_repair_count += 1

                    if final_attempt_record.passed:
                        recovered_by = "repair"
                        break

                    # After a failed repair, avoid repeated patching of semantic failures.
                    if current_fail == "TEST_FAIL":
                        if plan_available(history, policy_cfg):
                            forced_next_action = "plan"
                            break
                        stop_reason = "plan_budget_exhausted"
                        print(f"  stop: 🛑 {stop_reason}")
                        break

                    if current_fail == "EXEC_FAIL":
                        if same_signature_run_length(history) >= stagnation_threshold or not repair_available(total_repair_count, policy_cfg):
                            if plan_available(history, policy_cfg):
                                forced_next_action = "plan"
                                break
                            stop_reason = "plan_budget_exhausted"
                            print(f"  stop: 🛑 {stop_reason}")
                            break

                if stop_reason is not None or final_attempt_record.passed:
                    break

                if same_signature_run_length(history) >= stagnation_threshold or not repair_available(total_repair_count, policy_cfg):
                    if plan_available(history, policy_cfg):
                        forced_next_action = "plan"
                        continue
                    stop_reason = "plan_budget_exhausted"
                    print(f"  stop: 🛑 {stop_reason}")
                    break

                continue

        if final_exec_result is None:
            final_exec_result = final_attempt_record

        eval_results.append(final_exec_result)

        final_status = final_attempt_record.status
        failure_family = "PASS" if final_status == "PASS" else coarse_fail(final_status)

        initial_status = None
        initial_failure_family = None
        success_at_call = None

        for s in step_logs:
            if s["trajectory_id"] != trajectory_id:
                continue

            if s["stage"] == "generate":
                initial_status = s["status"]
                initial_failure_family = "PASS" if s["status"] == "PASS" else str(s["status"]).split(":")[0]

            if s["status"] == "PASS" and success_at_call is None:
                success_at_call = s["call_index"] + 1

        initial_failed = initial_status != "PASS"
        calls_to_recovery = (
            success_at_call - 1
            if initial_failed and success_at_call is not None
            else None
        )

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
            "used_plan": any(x["stage"] in {"plan", "replan"} for x in step_logs if x["trajectory_id"] == trajectory_id),
            "used_repair": any(x["stage"] == "repair" for x in step_logs if x["trajectory_id"] == trajectory_id),
            "recovered_by": recovered_by,
            "stopped_by_policy": stop_reason is not None and not final_attempt_record.passed,
            "stop_reason": stop_reason,
            "num_plan_calls": num_plan_calls(history),
            "num_repair_calls": total_repair_count,
            "budget_used": {
                "tokens": cumulative_total_tokens,
                "calls": call_count,
                "latency": cumulative_latency,
            },
            "input_token_stats": _compute_token_stats(problem_input_tokens),
            "output_token_stats": _compute_token_stats(problem_output_tokens),
            "initial_status": initial_status,
            "initial_failure_family": initial_failure_family,
            "initial_failed": initial_failed,
            "success_at_call": success_at_call,
            "calls_to_recovery": calls_to_recovery,
        }

        trajectory_logs.append(trajectory_entry)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # run-level token stats
    run_input_token_stats = _compute_token_stats(all_input_tokens)
    run_output_token_stats = _compute_token_stats(all_output_tokens)

    # ── 결과 요약 ──
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

    print(
        f"  input_tokens min/avg/max: "
        f"{run_input_token_stats['min']} / "
        f"{run_input_token_stats['avg']:.1f} / "
        f"{run_input_token_stats['max']}"
    )
    print(
        f"  output_tokens min/avg/max: "
        f"{run_output_token_stats['min']} / "
        f"{run_output_token_stats['avg']:.1f} / "
        f"{run_output_token_stats['max']}"
    )
    print(f"{'=' * 60}")

    total = len(trajectory_logs)

    # problem-level stats
    n_used_plan_problems = sum(1 for t in trajectory_logs if t.get("used_plan", False))
    n_plan_recovered_problems = sum(
        1 for t in trajectory_logs
        if t.get("recovered_by") in {"plan_code", "replan_code"}
    )

    n_used_repair_problems = sum(1 for t in trajectory_logs if t.get("used_repair", False))
    n_repair_recovered_problems = sum(
        1 for t in trajectory_logs
        if t.get("recovered_by") == "repair"
    )

    # call-level stats
    n_plan_calls = sum(1 for s in step_logs if s.get("stage") == "plan_code")
    n_plan_call_successes = sum(
        1 for s in step_logs
        if s.get("stage") == "plan_code" and s.get("status") == "PASS"
    )

    n_repair_calls = sum(1 for s in step_logs if s.get("stage") == "repair")
    n_repair_call_successes = sum(
        1 for s in step_logs
        if s.get("stage") == "repair" and s.get("status") == "PASS"
    )

    print(
        f"  [problem-level] plan 사용: {n_used_plan_problems}/{total} ({n_used_plan_problems/total*100:.1f}%)"
        if total > 0 else "  [problem-level] plan 사용: 0/0"
    )
    print(
        f"  [problem-level] plan 복구 성공: {n_plan_recovered_problems}/{n_used_plan_problems} ({n_plan_recovered_problems/n_used_plan_problems*100:.1f}%)"
        if n_used_plan_problems > 0 else "  [problem-level] plan 복구 성공: 0/0"
    )
    print(
        f"  [problem-level] repair 사용: {n_used_repair_problems}/{total} ({n_used_repair_problems/total*100:.1f}%)"
        if total > 0 else "  [problem-level] repair 사용: 0/0"
    )
    print(
        f"  [problem-level] repair 복구 성공: {n_repair_recovered_problems}/{n_used_repair_problems} ({n_repair_recovered_problems/n_used_repair_problems*100:.1f}%)"
        if n_used_repair_problems > 0 else "  [problem-level] repair 복구 성공: 0/0"
    )

    print(
        f"  [call-level] plan 호출 누적: {n_plan_calls}, 성공: {n_plan_call_successes}/{n_plan_calls} ({n_plan_call_successes/n_plan_calls*100:.1f}%)"
        if n_plan_calls > 0 else "  [call-level] plan 호출 누적: 0, 성공: 0/0"
    )
    print(
        f"  [call-level] repair 호출 누적: {n_repair_calls}, 성공: {n_repair_call_successes}/{n_repair_calls} ({n_repair_call_successes/n_repair_calls*100:.1f}%)"
        if n_repair_calls > 0 else "  [call-level] repair 호출 누적: 0, 성공: 0/0"
    )
    print(f"{'=' * 60}")

    if save_problem_summary:
        save_result(
            {
                "run_id": run_id,
                "dataset": dataset_name,
                "method": method_name,
                "total_problems": summary["total"],
                "num_pass": summary["passed"],
                "pass_at_1": summary["pass@1"],
                "execution_success_rate": summary["execution_success_rate"],
                "conditional_pass": summary["conditional_pass"],
                "extra_summary": extra_summary,
                "token_stats": {
                    "input": run_input_token_stats,
                    "output": run_output_token_stats,
                },
                "policy_stats": {
                    "problem_level": {
                        "plan_used_problems": n_used_plan_problems,
                        "plan_problem_usage_rate": n_used_plan_problems / total if total > 0 else 0.0,
                        "plan_recovered_problems": n_plan_recovered_problems,
                        "plan_problem_recovery_rate": (
                            n_plan_recovered_problems / n_used_plan_problems
                            if n_used_plan_problems > 0 else 0.0
                        ),
                        "repair_used_problems": n_used_repair_problems,
                        "repair_problem_usage_rate": n_used_repair_problems / total if total > 0 else 0.0,
                        "repair_recovered_problems": n_repair_recovered_problems,
                        "repair_problem_recovery_rate": (
                            n_repair_recovered_problems / n_used_repair_problems
                            if n_used_repair_problems > 0 else 0.0
                        ),
                    },
                    "call_level": {
                        "plan_call_count": n_plan_calls,
                        "plan_call_success_count": n_plan_call_successes,
                        "plan_call_success_rate": (
                            n_plan_call_successes / n_plan_calls
                            if n_plan_calls > 0 else 0.0
                        ),
                        "repair_call_count": n_repair_calls,
                        "repair_call_success_count": n_repair_call_successes,
                        "repair_call_success_rate": (
                            n_repair_call_successes / n_repair_calls
                            if n_repair_calls > 0 else 0.0
                        ),
                    },
                }
            },
            os.path.join(output_dir, "summary.json"),
        )

    if save_step_level:
        save_results_jsonl(step_logs, os.path.join(output_dir, "step_logs.jsonl"))
    if save_trajectory_level:
        save_results_jsonl(trajectory_logs, os.path.join(output_dir, "trajectory_logs.jsonl"))
    if save_run_analysis:
        save_result(
            {
                "run_id": run_id,
                "dataset": dataset_name,
                "method": method_name,
                "token_stats": {
                    "input": run_input_token_stats,
                    "output": run_output_token_stats,
                },
            },
            os.path.join(output_dir, "analysis.json"),
        )

    if save_failure_examples and failure_examples:
        save_result(failure_examples, os.path.join(output_dir, "failure_examples.json"))

    print("✅ policy_loop 완료")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.policy_loop <config.yaml>")
        sys.exit(1)

    run_policy_loop(sys.argv[1])