import gc
import os
import sys
import time
import yaml
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.models.hf_model import HFModel
from src.evaluation.metrics import (
    summarize_phase1_results,
    summarize_failure_breakdown,
)
from src.utils.io import save_result, save_results_jsonl, make_run_id
from src.utils.dataloader import load_task_and_adapter
from src.utils.prompt_loader import (
    build_planner_prompt_for_sample,
    build_coder_prompt_for_sample,
)
from src.utils.prompting.planner_coder import extract_planner_output


@dataclass
class PolicyState:
    dataset: str
    call_count: int = 0
    max_calls: int = 3

    exec_ok: bool = False
    test_pass: bool = False
    status: Optional[str] = None
    error_stage: Optional[str] = None   # "exec" | "test" | "unknown" | None
    error_type: Optional[str] = None
    tests_passed: int = 0
    tests_total: int = 0
    code_length: int = 0

    repeated_same_error: bool = False
    last_action: Optional[str] = None
    last_error_type: Optional[str] = None

    has_plan: bool = False
    action_history: List[str] = field(default_factory=list)

    def update_from_attempt(self, attempt_record) -> None:
        new_error_type = attempt_record.error_type

        self.repeated_same_error = (
            self.last_error_type is not None
            and new_error_type is not None
            and self.last_error_type == new_error_type
        )

        self.exec_ok = attempt_record.exec_ok
        self.test_pass = attempt_record.test_pass
        self.status = attempt_record.status
        self.error_stage = attempt_record.error_stage
        self.error_type = new_error_type
        self.last_error_type = new_error_type
        self.tests_passed = attempt_record.tests_passed or 0
        self.tests_total = attempt_record.tests_total or 0


ACTION_GENERATE = "generate"
ACTION_RETRY = "retry"
ACTION_REPAIR = "repair"
ACTION_PLAN = "plan"
ACTION_STOP = "stop"


def choose_action(state: PolicyState) -> tuple[str, str]:
    if state.call_count == 0:
        if state.has_plan:
            return ACTION_GENERATE, "첫 호출 (plan 보유) → 코드 생성"
        return ACTION_GENERATE, "첫 호출 → 코드 생성"

    if state.test_pass:
        return ACTION_STOP, "테스트 통과 → 종료"

    if state.call_count >= state.max_calls:
        return ACTION_STOP, f"예산 소진 ({state.call_count}/{state.max_calls}) → 종료"

    if (
        state.error_stage == "exec"
        and state.repeated_same_error
        and not state.has_plan
        and (state.max_calls - state.call_count) >= 2
    ):
        return ACTION_PLAN, "exec failure 반복 + plan 없음 → planner 호출"

    if state.error_stage == "exec":
        return ACTION_REPAIR, "exec failure → error message 기반 repair"

    if state.error_stage == "test" and state.repeated_same_error:
        return ACTION_REPAIR, "test failure 반복 → repair로 수정"

    if state.error_stage == "test":
        return ACTION_RETRY, "test failure → retry/refinement"

    return ACTION_REPAIR, "fallback → repair"


def _gen(model: HFModel, prompt: str, max_tokens: Optional[int] = None) -> dict:
    if max_tokens is not None:
        orig = model.max_new_tokens
        model.max_new_tokens = max_tokens
        try:
            result = model.generate(prompt)
        finally:
            model.max_new_tokens = orig
    else:
        result = model.generate(prompt)
    return result


def _run_codegen(
    *,
    sample,
    model: HFModel,
    adapter,
    prompt: str,
    method_name: str,
    model_name: str,
    attempt_idx: int,
    use_planner_extract: bool = False,
    max_tokens: Optional[int] = None,
) -> dict:
    t0 = time.perf_counter()
    gen_result = _gen(model, prompt, max_tokens)
    latency = time.perf_counter() - t0

    raw_text = gen_result["text"]

    if use_planner_extract:
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
        "gen_result": gen_result,
        "raw_text": raw_text,
        "generated_code": generated_code,
        "exec_result": exec_result,
        "attempt_record": attempt_record,
        "latency": latency,
    }


def _make_step_entry(
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
    code: Optional[str] = None,
    attempt_record=None,
    planner_text: Optional[str] = None,
    save_code: bool = True,
    sample=None,
    policy_action: Optional[str] = None,
    policy_reason: Optional[str] = None,
    policy_state_snapshot: Optional[dict] = None,
) -> dict:
    is_planner = stage == "plan"
    is_repair = stage == "repair"
    is_retry = stage == "retry"

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
        "is_retry": is_retry,
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
        "policy_action": policy_action,
        "policy_reason": policy_reason,
        "policy_state": policy_state_snapshot,
    }

    if planner_text is not None:
        entry["plan_text"] = planner_text

    if hasattr(sample, "entry_point"):
        entry["entry_point"] = sample.entry_point

    return entry


def run_adaptive_policy(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg = config.get("run", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    method_cfg = config.get("method", {})
    budget_cfg = config.get("budget", {})
    output_cfg = config.get("output", {})
    logging_cfg = config.get("logging", {})
    policy_cfg = config.get("policy", {})

    run_id = make_run_id(config)
    seed = run_cfg.get("seed", 42)
    dataset_name = dataset_cfg.get("name", "humaneval")
    num_samples = dataset_cfg.get("num_samples", 1)

    model_name = model_cfg.get("name", "")
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    temperature = model_cfg.get("temperature", 0.0)

    planner_cfg = model_cfg.get("planner", {})
    planner_name = planner_cfg.get("name", model_name)
    planner_tokens = planner_cfg.get("max_new_tokens", 256)
    planner_temperature = planner_cfg.get("temperature", temperature)

    method_name = method_cfg.get("name", "adaptive_policy")
    max_calls = budget_cfg.get("max_calls", 4)

    allow_plan = policy_cfg.get("allow_plan", True)
    use_planner_extract = policy_cfg.get("use_planner_extract", False)

    output_dir = output_cfg.get(
        "dir", f"results/phase1_ver3/{dataset_name}/adaptive_policy"
    )

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Adaptive Policy 실험")
    print("=" * 60)
    print(f"run_id         : {run_id}")
    print(f"dataset        : {dataset_name}")
    print(f"method         : {method_name}")
    print(f"max_calls      : {max_calls}")
    print(f"allow_plan     : {allow_plan}")
    print(f"model          : {model_name}")
    print(f"max_new_tokens : {max_new_tokens}")
    print(f"temperature    : {temperature}")
    print(f"seed           : {seed}")
    print(f"output_dir     : {output_dir}")
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
            "policy": policy_cfg,
            "config_path": config_path,
        },
        os.path.join(output_dir, "config.json"),
    )

    task, adapter = load_task_and_adapter(dataset_name)
    print(f"dataset: {dataset_name} | size={len(task)}")

    main_model = HFModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if allow_plan:
        if planner_name == model_name:
            planner_model = main_model
        else:
            planner_model = HFModel(
                model_name=planner_name,
                max_new_tokens=planner_tokens,
                temperature=planner_temperature,
            )
    else:
        planner_model = None

    step_logs = []
    trajectory_logs = []
    eval_results = []

    samples_to_run = min(num_samples, len(task))
    global_step_id = 0

    for i in range(samples_to_run):
        sample = task.get_sample(i)
        problem_id = sample.task_id
        trajectory_id = f"{problem_id}_run0"

        cumulative_total_tokens = 0
        cumulative_latency = 0.0
        transition_path = []
        num_exec_fail = 0
        num_test_fail = 0
        call_count = 0

        previous_code = None
        error_message = None
        planner_output = None
        last_attempt = None
        final_exec_result = None

        state = PolicyState(
            dataset=dataset_name,
            call_count=0,
            max_calls=max_calls,
        )

        while True:
            state_snapshot = {
                "call_count": state.call_count,
                "exec_ok": state.exec_ok,
                "test_pass": state.test_pass,
                "status": state.status,
                "error_stage": state.error_stage,
                "error_type": state.error_type,
                "tests_passed": state.tests_passed,
                "tests_total": state.tests_total,
                "code_length": state.code_length,
                "repeated_same_error": state.repeated_same_error,
                "last_action": state.last_action,
                "has_plan": state.has_plan,
            }

            action, reason = choose_action(state)

            if action == ACTION_PLAN and not allow_plan:
                action = ACTION_REPAIR
                reason += " (plan disabled -> repair)"

            if action == ACTION_STOP:
                break

            if action == ACTION_PLAN:
                planner_prompt = build_planner_prompt_for_sample(sample)

                t0 = time.perf_counter()
                plan_gen = _gen(planner_model, planner_prompt, planner_tokens)
                plan_latency = time.perf_counter() - t0

                planner_output = extract_planner_output(plan_gen["text"])
                state.has_plan = True

                call_count += 1
                cumulative_total_tokens += plan_gen["total_tokens"]
                cumulative_latency += plan_latency
                state.call_count = call_count
                transition_path.append("PLAN_DONE")

                plan_entry = _make_step_entry(
                    run_id=run_id,
                    dataset_name=dataset_name,
                    problem_id=problem_id,
                    method_name=method_name,
                    trajectory_id=trajectory_id,
                    step_id=global_step_id,
                    call_index=call_count - 1,
                    stage="plan",
                    gen_result=plan_gen,
                    latency_sec=plan_latency,
                    planner_text=planner_output,
                    save_code=save_code,
                    sample=sample,
                    policy_action=action,
                    policy_reason=reason,
                    policy_state_snapshot=state_snapshot,
                )
                step_logs.append(plan_entry)

                global_step_id += 1
                state.last_action = ACTION_PLAN
                state.action_history.append(ACTION_PLAN)
                continue

            if action == ACTION_GENERATE:
                if state.has_plan and planner_output:
                    prompt = build_coder_prompt_for_sample(sample, planner_output)
                else:
                    prompt = adapter.build_initial_prompt(sample)
                stage = "generate"

            elif action == ACTION_RETRY:
                prompt = adapter.build_refinement_prompt(
                    sample=sample,
                    previous_code=previous_code,
                )
                stage = "retry"

            elif action == ACTION_REPAIR:
                prompt = adapter.build_repair_prompt(
                    sample=sample,
                    previous_code=previous_code,
                    error_message=error_message,
                )
                stage = "repair"

            else:
                raise ValueError(f"Unknown action: {action}")

            attempt = _run_codegen(
                sample=sample,
                model=main_model,
                adapter=adapter,
                prompt=prompt,
                method_name=method_name,
                model_name=model_name,
                attempt_idx=call_count,
                use_planner_extract=use_planner_extract and state.has_plan,
                max_tokens=None,
            )

            call_count += 1
            cumulative_total_tokens += attempt["gen_result"]["total_tokens"]
            cumulative_latency += attempt["latency"]
            state.call_count = call_count

            state.update_from_attempt(attempt["attempt_record"])
            state.code_length = len(attempt["generated_code"]) if attempt["generated_code"] else 0
            state.last_action = action
            state.action_history.append(action)

            current_status = attempt["attempt_record"].status
            transition_path.append(current_status)

            if str(current_status).startswith("EXEC_FAIL"):
                num_exec_fail += 1
            if str(current_status).startswith("TEST_FAIL"):
                num_test_fail += 1

            previous_code = attempt["generated_code"]
            error_message = attempt["attempt_record"].error_message
            last_attempt = attempt
            final_exec_result = attempt["exec_result"]

            step_entry = _make_step_entry(
                run_id=run_id,
                dataset_name=dataset_name,
                problem_id=problem_id,
                method_name=method_name,
                trajectory_id=trajectory_id,
                step_id=global_step_id,
                call_index=call_count - 1,
                stage=stage,
                gen_result=attempt["gen_result"],
                latency_sec=attempt["latency"],
                code=attempt["generated_code"],
                attempt_record=attempt["attempt_record"],
                save_code=save_code,
                sample=sample,
                policy_action=action,
                policy_reason=reason,
                policy_state_snapshot=state_snapshot,
            )
            step_logs.append(step_entry)
            global_step_id += 1

        if last_attempt is None:
            continue

        eval_results.append(final_exec_result)

        final_attempt_record = last_attempt["attempt_record"]
        final_status = final_attempt_record.status
        failure_family = "PASS" if final_status == "PASS" else str(final_status).split(":")[0]

        trajectory_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "num_steps": len(transition_path),
            "call_count": call_count,
            "final_status": final_status,
            "failure_family": failure_family,
            "final_tests_passed": final_attempt_record.tests_passed,
            "final_tests_total": final_attempt_record.tests_total,
            "total_tokens": cumulative_total_tokens,
            "total_latency": cumulative_latency,
            "num_exec_fail": num_exec_fail,
            "num_test_fail": num_test_fail,
            "transition_path": transition_path,
            "action_history": state.action_history,
            "budget_used": {
                "tokens": cumulative_total_tokens,
                "calls": call_count,
                "latency": cumulative_latency,
            },
        }
        trajectory_logs.append(trajectory_entry)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del trajectory_entry, last_attempt, final_exec_result, state
        gc.collect()

    summary = summarize_phase1_results(eval_results)

    extra_summary = summarize_failure_breakdown(eval_results)

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
            family = str(status).split(":")[0]
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.adaptive_policy <config.yaml>")
        sys.exit(1)

    run_adaptive_policy(sys.argv[1])