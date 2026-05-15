"""
Policy-main

stage별 오류 유형과, 실패 반복 패턴을 기반으로, repair / replan 중 적절한 전략을 동적으로 선택하는 policy를 구현하고자 함.

MAX_CALLS = 20

Generate = 1 LM call
Repair = 1 LM call
Planning = 2 LM calls
Terminate = 0 LM calls

MAX_PLAN_CYCLES = 3        # Planning 3회 = 6 LM calls
MAX_REPAIR_CALLS = 6       # Repair 전체 최대 6 LM calls (global budget)

STAGNATION_K = 3
NAMEERROR_K = 2
PLANNING_FAILURE_LIMIT = 2

Generate: 1 call
Planning: 최대 3 cycles = 6 calls
Repair: 최대 6 calls

총 명시적 제어 budget = 13 calls

Policy Rules:
    1. Generate if first step
    2. Terminate if PASS
    3. Terminate if call_count >= 20
    4. Compute planning_available / repair_available / planning_available_for_state
    5. Update planning_fail_count_by_state
    6. NameError repeated after planning → Terminate
    7. Previous action was Planning → Repair if possible, else Terminate
    8. Stagnation → Planning if possible, else Repair if possible, else Terminate
    9. High-value failure → Planning if possible
    10. Default → Repair if possible, else Terminate
"""
from __future__ import annotations

import gc
import os
import sys
import time
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from src.models.hf_model_vllm import HFModel
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
# Policy Constants
# ============================================================

MAX_CALLS = 20
MAX_PLAN_CYCLES = 3
MAX_REPAIR_CALLS = 6
PLANNING_FAILURE_LIMIT = 2
STAGNATION_K = 3
NAMEERROR_K = 2

HIGH_VALUE_ERRORS = {
    "AssertionError",
    "SyntaxError",
    "NameError",
}


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
# Policy State
# ============================================================

@dataclass
class PolicyState:
    """Per-problem mutable policy state."""
    call_count: int = 0
    plan_cycle_count: int = 0
    repair_call_count: int = 0

    previous_action: Optional[str] = None          # "generate" / "repair" / "plan"
    previous_failure_state: Optional[Tuple] = None  # (failure_type, error_type)

    same_failure_state_repeat_count: int = 0
    same_error_repeat_count: int = 0

    planning_fail_count_by_state: Dict[Tuple, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def get_failure_state(self, attempt_record) -> Tuple[str, str]:
        ft = coarse_fail(attempt_record.status)
        et = normalize_error_type(attempt_record.error_type)
        return (ft, et)

    def update_repeat_counts(self, current_failure_state: Tuple):
        if current_failure_state == self.previous_failure_state:
            self.same_failure_state_repeat_count += 1
        else:
            self.same_failure_state_repeat_count = 1

        cur_error = current_failure_state[1]
        prev_error = self.previous_failure_state[1] if self.previous_failure_state else None
        if cur_error == prev_error:
            self.same_error_repeat_count += 1
        else:
            self.same_error_repeat_count = 1

    def select_action(self, attempt_record) -> str:
        """
        Main policy decision function.

        Rules (in order):
            1. Generate if first step
            2. Terminate if PASS
            3. Terminate if call_count >= MAX_CALLS
            4. Compute availability flags
            5. Update planning_fail_count_by_state
            6. NameError repeated after planning → Terminate
            7. Previous action was Planning → Repair if possible, else Terminate
            8. Stagnation → Planning / Repair / Terminate
            9. High-value failure → Planning if possible
            10. Default → Repair / Terminate
        """
        # --------------------------------------------------
        # Rule 1. Initial Rule
        # --------------------------------------------------
        if self.call_count == 0:
            return "generate"

        # --------------------------------------------------
        # Rule 2. Success Rule
        # --------------------------------------------------
        if attempt_record.passed:
            return "terminate"

        # --------------------------------------------------
        # Rule 3. Global Budget Rule
        # --------------------------------------------------
        if self.call_count >= MAX_CALLS:
            return "terminate"

        current_failure_state = self.get_failure_state(attempt_record)
        error_type = normalize_error_type(attempt_record.error_type)

        # --------------------------------------------------
        # Rule 4. Compute availability flags
        # --------------------------------------------------
        planning_available = (
            self.call_count + 2 <= MAX_CALLS
            and self.plan_cycle_count < MAX_PLAN_CYCLES
        )

        repair_available = (
            self.call_count + 1 <= MAX_CALLS
            and self.repair_call_count < MAX_REPAIR_CALLS
        )

        planning_available_for_state = (
            self.planning_fail_count_by_state[current_failure_state]
            < PLANNING_FAILURE_LIMIT
        )

        # --------------------------------------------------
        # Rule 5. Update planning_fail_count_by_state
        # --------------------------------------------------
        if (
            self.previous_action == "plan"
            and current_failure_state == self.previous_failure_state
        ):
            self.planning_fail_count_by_state[current_failure_state] += 1

        # --------------------------------------------------
        # Rule 6. NameError Termination Rule
        # --------------------------------------------------
        if (
            error_type == "NameError"
            and self.same_error_repeat_count >= NAMEERROR_K
            and self.plan_cycle_count > 0
        ):
            return "terminate"

        # --------------------------------------------------
        # Rule 7. Post-Planning Repair Rule
        # --------------------------------------------------
        if self.previous_action == "plan":
            if repair_available:
                return "repair"
            else:
                return "terminate"

        # --------------------------------------------------
        # Rule 8. Stagnation Rule
        # --------------------------------------------------
        if self.same_failure_state_repeat_count >= STAGNATION_K:
            if planning_available and planning_available_for_state:
                return "plan"
            elif repair_available:
                return "repair"
            else:
                return "terminate"

        # --------------------------------------------------
        # Rule 9. High-value Planning Rule
        # --------------------------------------------------
        if (
            error_type in HIGH_VALUE_ERRORS
            and planning_available
            and planning_available_for_state
        ):
            return "plan"

        # --------------------------------------------------
        # Rule 10. Default Repair Rule
        # --------------------------------------------------
        if repair_available:
            return "repair"

        return "terminate"

    def after_action(self, action: str, attempt_record):
        """Call after executing an action to update counters and previous state."""
        current_failure_state = self.get_failure_state(attempt_record)

        if action == "generate":
            self.call_count += 1
        elif action == "repair":
            self.call_count += 1
            self.repair_call_count += 1
        elif action == "plan":
            self.call_count += 2   # planner + coder
            self.plan_cycle_count += 1

        self.update_repeat_counts(current_failure_state)
        self.previous_failure_state = current_failure_state
        self.previous_action = action


# ============================================================
# Policy helpers (shared util)
# ============================================================

def coarse_fail(status: str) -> str:
    if status == "PASS":
        return "PASS"
    return str(status).split(":")[0]


def normalize_error_type(error_type: Optional[str]) -> str:
    if error_type is None:
        return "NONE"
    s = str(error_type).strip()
    return s if s else "NONE"


# ============================================================
# Step runners  (identical to policy_loop.py)
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

def _append_step_log(
    step_logs: List[Dict[str, Any]],
    *,
    save_step_level: bool,
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
    if not save_step_level:
        return

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


def _make_policy_state_snapshot(ps: PolicyState) -> Dict[str, Any]:
    return {
        "call_count": ps.call_count,
        "plan_cycle_count": ps.plan_cycle_count,
        "repair_call_count": ps.repair_call_count,
        "previous_action": ps.previous_action,
        "previous_failure_state": ps.previous_failure_state,
        "same_failure_state_repeat_count": ps.same_failure_state_repeat_count,
        "same_error_repeat_count": ps.same_error_repeat_count,
        "planning_fail_count_by_state": dict(ps.planning_fail_count_by_state),
    }


# ============================================================
# Main runner
# ============================================================

def run_policy_loop(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg      = config.get("run", {})
    dataset_cfg  = config.get("dataset", {})
    model_cfg    = config.get("model", {})
    method_cfg   = config.get("method", {})
    budget_cfg   = config.get("budget", {})
    output_cfg   = config.get("output", {})
    logging_cfg  = config.get("logging", {})
    debug_cfg    = config.get("debug", {})

    run_id   = make_run_id(config)
    seed     = run_cfg.get("seed", 42)

    dataset_name = dataset_cfg.get("name", "humaneval")
    num_samples  = dataset_cfg.get("num_samples", 1)

    model_name     = model_cfg.get("name", "")
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    temperature    = model_cfg.get("temperature", 0.0)

    planner_cfg_inner  = model_cfg.get("planner", {})
    planner_max_tokens = planner_cfg_inner.get("max_new_tokens", 256)

    coder_cfg_inner  = model_cfg.get("coder", {})
    coder_max_tokens = coder_cfg_inner.get("max_new_tokens", max_new_tokens)

    repair_cfg_inner  = model_cfg.get("repair", {})
    repair_max_tokens = repair_cfg_inner.get("max_new_tokens", max_new_tokens)

    method_name = method_cfg.get("name", "policy_loop_main")
    max_calls   = budget_cfg.get("max_calls", MAX_CALLS)

    output_dir = output_cfg.get("dir", f"results/RUN/{dataset_name}/policy_loop_main")

    save_step_level      = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary  = logging_cfg.get("save_problem_summary", True)
    save_run_analysis     = logging_cfg.get("save_run_analysis", True)
    save_code             = logging_cfg.get("save_code", True)
    save_failure_examples = logging_cfg.get("failure_examples", True)

    debug_mode = debug_cfg.get("mode", "run")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🔄 Policy-main Loop 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"model               : {model_name}")
    print(f"max_new_tokens      : {max_new_tokens}")
    print(f"planner_max_tokens  : {planner_max_tokens}")
    print(f"repair_max_tokens   : {repair_max_tokens}")
    print(f"max_calls           : {max_calls}")
    print(f"MAX_PLAN_CYCLES     : {MAX_PLAN_CYCLES}")
    print(f"MAX_REPAIR_CALLS    : {MAX_REPAIR_CALLS}")
    print(f"STAGNATION_K        : {STAGNATION_K}")
    print(f"NAMEERROR_K         : {NAMEERROR_K}")
    print(f"PLANNING_FAILURE_LIMIT : {PLANNING_FAILURE_LIMIT}")
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
            "policy_constants": {
                "MAX_CALLS": MAX_CALLS,
                "MAX_PLAN_CYCLES": MAX_PLAN_CYCLES,
                "MAX_REPAIR_CALLS": MAX_REPAIR_CALLS,
                "PLANNING_FAILURE_LIMIT": PLANNING_FAILURE_LIMIT,
                "STAGNATION_K": STAGNATION_K,
                "NAMEERROR_K": NAMEERROR_K,
                "HIGH_VALUE_ERRORS": list(HIGH_VALUE_ERRORS),
            },
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

    step_logs: List[Dict[str, Any]] = []
    trajectory_logs: List[Dict[str, Any]] = []
    eval_results = []
    failure_examples = {}

    samples_to_run = min(num_samples, len(task))

    for i in range(samples_to_run):
        sample     = task.get_sample(i)
        problem_id = sample.task_id
        trajectory_id = f"{problem_id}_run0"

        print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

        # ── per-problem accumulators ──
        ps = PolicyState()

        cumulative_input_tokens  = 0
        cumulative_output_tokens = 0
        cumulative_total_tokens  = 0
        cumulative_latency       = 0.0

        num_exec_fail = 0
        num_test_fail = 0
        transition_path: List[str] = []

        final_attempt_record = None
        final_exec_result    = None
        recovered_by         = None
        stop_reason          = None

        previous_code   = None
        planner_output  = None

        step_id     = 0
        attempt_idx = 0

        initial_prompt = adapter.build_initial_prompt(sample)

        # ──────────────────────────────────────────────────
        # Step 0: Generate
        # ──────────────────────────────────────────────────
        try:
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
        except Exception as e:
            print(f"  ⚠️ 모델 호출 실패 (토큰 초과 등), 스킵: {e}")
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
                "final_status": "TOKEN_OVERFLOW",
                "failure_family": "TOKEN_OVERFLOW",
                "final_tests_passed": 0, "final_tests_total": 0,
                "total_tokens": 0, "total_latency": 0,
                "num_exec_fail": 0, "num_test_fail": 0,
                "transition_path": ["TOKEN_OVERFLOW"],
                "budget_used": {"tokens": 0, "calls": 0, "latency": 0},
            })
            continue

        cumulative_input_tokens  += gen["input_tokens"]
        cumulative_output_tokens += gen["output_tokens"]
        cumulative_total_tokens  += gen["total_tokens"]
        cumulative_latency       += gen["latency_sec"]

        final_attempt_record = gen["attempt_record"]
        final_exec_result    = gen["exec_result"]
        previous_code        = gen["generated_code"]

        current_fail  = coarse_fail(final_attempt_record.status)
        current_error = normalize_error_type(final_attempt_record.error_type)
        transition_path.append(f"{current_fail}:{current_error}")

        if current_fail == "EXEC_FAIL":
            num_exec_fail += 1
        if current_fail == "TEST_FAIL":
            num_test_fail += 1

        # update policy state after generate
        ps.after_action("generate", final_attempt_record)

        _append_step_log(
            step_logs,
            save_step_level=save_step_level,
            run_id=run_id,
            dataset_name=dataset_name,
            problem_id=problem_id,
            method_name=method_name,
            trajectory_id=trajectory_id,
            step_id=step_id,
            call_index=ps.call_count - 1,
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

        pretty = "✅ PASS" if final_attempt_record.passed else f"❌ {final_attempt_record.status}"
        print(f"  generate: {pretty}")

        _collect_failure_example(
            failure_examples,
            problem_id=problem_id,
            attempt_idx=attempt_idx,
            status=final_attempt_record.status,
            prompt=initial_prompt,
            raw_text=gen["raw_text"],
            generated_code=gen["generated_code"],
            error_type=final_attempt_record.error_type,
            error_stage=final_attempt_record.error_stage,
            error_message=final_attempt_record.error_message,
        )

        attempt_idx += 1
        step_id     += 1

        if final_attempt_record.passed:
            recovered_by = "generate"

        # ──────────────────────────────────────────────────
        # Main refinement loop
        # ──────────────────────────────────────────────────
        while not final_attempt_record.passed:

            action = ps.select_action(final_attempt_record)

            if action == "terminate":
                stop_reason = (
                    "max_calls_reached"
                    if ps.call_count >= max_calls
                    else "policy_terminate"
                )
                print(f"  stop: 🛑 {stop_reason}")
                break

            # ── Plan ──────────────────────────────────────
            if action == "plan":
                try:
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
                except Exception as e:
                    print(f"  ⚠️ plan 모델 호출 실패 (토큰 초과 등), 이 문제 중단: {e}")
                    stop_reason = "token_overflow"
                    transition_path.append("TOKEN_OVERFLOW")
                    break

                # planner call log
                cumulative_input_tokens  += out["plan_input_tokens"]
                cumulative_output_tokens += out["plan_output_tokens"]
                cumulative_total_tokens  += out["plan_total_tokens"]
                cumulative_latency       += out["plan_latency"]

                _append_step_log(
                    step_logs,
                    save_step_level=save_step_level,
                    run_id=run_id,
                    dataset_name=dataset_name,
                    problem_id=problem_id,
                    method_name=method_name,
                    trajectory_id=trajectory_id,
                    step_id=step_id,
                    call_index=ps.call_count,          # before after_action
                    stage="plan",
                    policy_action="plan",
                    policy_state=_make_policy_state_snapshot(ps),
                    policy_stop=False,
                    policy_stop_reason=None,
                    sample=sample,
                    save_code=save_code,
                    generated_code=None,
                    attempt_record=SimpleNamespace(
                        status="PLAN_DONE",
                        exec_ok=None, test_pass=None,
                        error_type=None, error_stage=None,
                        error_message=None,
                        tests_passed=None, tests_total=None,
                    ),
                    input_tokens=out["plan_input_tokens"],
                    output_tokens=out["plan_output_tokens"],
                    total_tokens=out["plan_total_tokens"],
                    latency_sec=out["plan_latency"],
                    planner_output=out["planner_output"],
                    is_planner=True,
                )
                transition_path.append("PLAN_DONE")
                step_id += 1

                # coder call log
                cumulative_input_tokens  += out["input_tokens"]
                cumulative_output_tokens += out["output_tokens"]
                cumulative_total_tokens  += out["total_tokens"]
                cumulative_latency       += out["latency_sec"]

                final_attempt_record = out["attempt_record"]
                final_exec_result    = out["exec_result"]
                previous_code        = out["generated_code"]
                planner_output       = out["planner_output"]

                current_fail  = coarse_fail(final_attempt_record.status)
                current_error = normalize_error_type(final_attempt_record.error_type)
                transition_path.append(f"{current_fail}:{current_error}")

                if current_fail == "EXEC_FAIL":
                    num_exec_fail += 1
                if current_fail == "TEST_FAIL":
                    num_test_fail += 1

                ps.after_action("plan", final_attempt_record)

                _append_step_log(
                    step_logs,
                    save_step_level=save_step_level,
                    run_id=run_id,
                    dataset_name=dataset_name,
                    problem_id=problem_id,
                    method_name=method_name,
                    trajectory_id=trajectory_id,
                    step_id=step_id,
                    call_index=ps.call_count - 1,
                    stage="plan_code",
                    policy_action="plan",
                    policy_state=_make_policy_state_snapshot(ps),
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

                pretty = "✅ PASS" if final_attempt_record.passed else f"❌ {final_attempt_record.status}"
                print(f"  plan_code: {pretty}")

                _collect_failure_example(
                    failure_examples,
                    problem_id=problem_id,
                    attempt_idx=attempt_idx,
                    status=final_attempt_record.status,
                    prompt=out["coder_prompt"],
                    raw_text=out["raw_text"],
                    generated_code=out["generated_code"],
                    error_type=final_attempt_record.error_type,
                    error_stage=final_attempt_record.error_stage,
                    error_message=final_attempt_record.error_message,
                )
                attempt_idx += 1

                if final_attempt_record.passed:
                    recovered_by = "plan_code"

            # ── Repair ────────────────────────────────────
            elif action == "repair":
                try:
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
                except Exception as e:
                    print(f"  ⚠️ repair 모델 호출 실패 (토큰 초과 등), 이 문제 중단: {e}")
                    stop_reason = "token_overflow"
                    transition_path.append("TOKEN_OVERFLOW")
                    break

                cumulative_input_tokens  += out["input_tokens"]
                cumulative_output_tokens += out["output_tokens"]
                cumulative_total_tokens  += out["total_tokens"]
                cumulative_latency       += out["latency_sec"]

                final_attempt_record = out["attempt_record"]
                final_exec_result    = out["exec_result"]
                previous_code        = out["generated_code"]

                current_fail  = coarse_fail(final_attempt_record.status)
                current_error = normalize_error_type(final_attempt_record.error_type)
                transition_path.append(f"{current_fail}:{current_error}")

                if current_fail == "EXEC_FAIL":
                    num_exec_fail += 1
                if current_fail == "TEST_FAIL":
                    num_test_fail += 1

                ps.after_action("repair", final_attempt_record)

                _append_step_log(
                    step_logs,
                    save_step_level=save_step_level,
                    run_id=run_id,
                    dataset_name=dataset_name,
                    problem_id=problem_id,
                    method_name=method_name,
                    trajectory_id=trajectory_id,
                    step_id=step_id,
                    call_index=ps.call_count - 1,
                    stage="repair",
                    policy_action="repair",
                    policy_state=_make_policy_state_snapshot(ps),
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
                    is_retry=ps.repair_call_count > 1,
                )
                step_id += 1

                pretty = "✅ PASS" if final_attempt_record.passed else f"❌ {final_attempt_record.status}"
                print(f"  repair [{ps.repair_call_count}/{MAX_REPAIR_CALLS}]: {pretty}")

                _collect_failure_example(
                    failure_examples,
                    problem_id=problem_id,
                    attempt_idx=attempt_idx,
                    status=final_attempt_record.status,
                    prompt=out["prompt"],
                    raw_text=out["raw_text"],
                    generated_code=out["generated_code"],
                    error_type=final_attempt_record.error_type,
                    error_stage=final_attempt_record.error_stage,
                    error_message=final_attempt_record.error_message,
                )
                attempt_idx += 1

                if final_attempt_record.passed:
                    recovered_by = "repair"

        # ── end while ─────────────────────────────────────

        if final_exec_result is None:
            final_exec_result = final_attempt_record

        setattr(final_exec_result, "num_calls", ps.call_count)
        eval_results.append(final_exec_result)

        final_status   = final_attempt_record.status
        failure_family = "PASS" if final_status == "PASS" else coarse_fail(final_status)

        trajectory_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "num_steps": ps.call_count,
            "call_count": ps.call_count,
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
            "used_plan":   any(s["stage"] == "plan"   for s in step_logs if s["trajectory_id"] == trajectory_id),
            "used_repair": any(s["stage"] == "repair" for s in step_logs if s["trajectory_id"] == trajectory_id),
            "recovered_by": recovered_by,
            "stopped_by_policy": stop_reason is not None and not final_attempt_record.passed,
            "stop_reason": stop_reason,
            "num_plan_calls": ps.plan_cycle_count,
            "num_repair_calls": ps.repair_call_count,
            "budget_used": {
                "tokens": cumulative_total_tokens,
                "calls": ps.call_count,
                "latency": cumulative_latency,
            },
            "planning_cycle_count": ps.plan_cycle_count,
            "planning_cycle_call_cost": ps.plan_cycle_count * 2,
            "repair_call_count": ps.repair_call_count,
        }

        if save_trajectory_level:
            trajectory_logs.append(trajectory_entry)

        gc.collect()

    # ── Summary ───────────────────────────────────────────
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

    total = len(trajectory_logs)

    n_used_plan_problems   = sum(1 for t in trajectory_logs if t.get("used_plan", False))
    n_plan_recovered       = sum(1 for t in trajectory_logs if t.get("recovered_by") == "plan_code")
    n_used_repair_problems = sum(1 for t in trajectory_logs if t.get("used_repair", False))
    n_repair_recovered     = sum(1 for t in trajectory_logs if t.get("recovered_by") == "repair")

    n_planning_cycles         = sum(1 for s in step_logs if s.get("stage") == "plan")
    n_planning_code_successes = sum(1 for s in step_logs if s.get("stage") == "plan_code" and s.get("status") == "PASS")
    planning_cycle_call_cost  = n_planning_cycles * 2
    planning_cycle_success_rate = (n_planning_code_successes / n_planning_cycles) if n_planning_cycles > 0 else 0.0

    n_repair_calls         = sum(1 for s in step_logs if s.get("stage") == "repair")
    n_repair_call_successes = sum(1 for s in step_logs if s.get("stage") == "repair" and s.get("status") == "PASS")

    def _pct(a, b): return f"{a/b*100:.1f}%" if b > 0 else "0/0"

    print(f"  [problem] plan 사용:     {n_used_plan_problems}/{total} ({_pct(n_used_plan_problems, total)})")
    print(f"  [problem] plan 복구:     {n_plan_recovered}/{n_used_plan_problems} ({_pct(n_plan_recovered, n_used_plan_problems)})")
    print(f"  [problem] repair 사용:   {n_used_repair_problems}/{total} ({_pct(n_used_repair_problems, total)})")
    print(f"  [problem] repair 복구:   {n_repair_recovered}/{n_used_repair_problems} ({_pct(n_repair_recovered, n_used_repair_problems)})")
    print(f"  [call] planning cycles:  {n_planning_cycles} ({planning_cycle_call_cost} calls) "
          f"성공 {n_planning_code_successes}/{n_planning_cycles} ({_pct(n_planning_code_successes, n_planning_cycles)})")
    print(f"  [call] repair calls:     {n_repair_calls} "
          f"성공 {n_repair_call_successes}/{n_repair_calls} ({_pct(n_repair_call_successes, n_repair_calls)})")
    print(f"{'=' * 60}")

    if save_problem_summary:
        save_result(
            {
                "run_id": run_id,
                "dataset": dataset_name,
                "method": method_name,
                "total_problems": summary["total"],
                "max_calls": max_calls,
                "num_success": summary["success"],
                "success_metric_name": success_key,
                "success_at_k": summary[success_key],
                "success_at_k_curve": summary["success_at_k_curve"],
                "ausc": summary["ausc"],
                "execution_success_rate": summary["execution_success_rate"],
                "conditional_success": summary["conditional_success"],
                "extra_summary": extra_summary,
                "policy_constants": {
                    "MAX_CALLS": MAX_CALLS,
                    "MAX_PLAN_CYCLES": MAX_PLAN_CYCLES,
                    "MAX_REPAIR_CALLS": MAX_REPAIR_CALLS,
                    "PLANNING_FAILURE_LIMIT": PLANNING_FAILURE_LIMIT,
                    "STAGNATION_K": STAGNATION_K,
                    "NAMEERROR_K": NAMEERROR_K,
                },
                "policy_stats": {
                    "problem_level": {
                        "plan_used_problems": n_used_plan_problems,
                        "plan_problem_usage_rate": n_used_plan_problems / total if total > 0 else 0.0,
                        "plan_recovered_problems": n_plan_recovered,
                        "plan_problem_recovery_rate": n_plan_recovered / n_used_plan_problems if n_used_plan_problems > 0 else 0.0,
                        "repair_used_problems": n_used_repair_problems,
                        "repair_problem_usage_rate": n_used_repair_problems / total if total > 0 else 0.0,
                        "repair_recovered_problems": n_repair_recovered,
                        "repair_problem_recovery_rate": n_repair_recovered / n_used_repair_problems if n_used_repair_problems > 0 else 0.0,
                    },
                    "call_level": {
                        "planning_cycle_count": n_planning_cycles,
                        "planning_cycle_call_cost": planning_cycle_call_cost,
                        "planning_code_success_count": n_planning_code_successes,
                        "planning_cycle_success_rate": planning_cycle_success_rate,
                        "repair_call_count": n_repair_calls,
                        "repair_call_success_count": n_repair_call_successes,
                        "repair_call_success_rate": n_repair_call_successes / n_repair_calls if n_repair_calls > 0 else 0.0,
                    },
                },
            },
            os.path.join(output_dir, "summary.json"),
        )

    if save_step_level:
        save_results_jsonl(step_logs, os.path.join(output_dir, "step_logs.jsonl"))
    if save_trajectory_level:
        save_results_jsonl(trajectory_logs, os.path.join(output_dir, "trajectory_logs.jsonl"))
    if save_run_analysis:
        save_result(
            {"run_id": run_id, "dataset": dataset_name, "method": method_name},
            os.path.join(output_dir, "analysis.json"),
        )
    if save_failure_examples and failure_examples:
        save_result(failure_examples, os.path.join(output_dir, "failure_examples.json"))

    print("✅ policy_loop_main 완료")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.policy_loop_main <config.yaml>")
        sys.exit(1)

    run_policy_loop(sys.argv[1])