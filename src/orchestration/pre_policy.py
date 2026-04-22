"""
현재 상태(PolicyState)를 보고 generate / retry / repair / plan / stop 중 하나를 선택해서 orchestration을 진행하는 상위 정책 모듈.

1단계  choose_action(state) : 규칙 기반 action 결정 함수
2단계  run_adaptive_policy  : unified runner (단일 루프에서 모든 action 처리)
3단계  per-step 로그        : state / action / action_reason을 step_logs에 기록
4단계  비교 실험 준비       : method="adaptive_policy"로 저장, 기존 결과와 동일한 포맷 유지 → 외부 분석 스크립트가 바로 사용 가능

step-level 로그 필드:
stage / call_index / exec_ok / test_pass / status /
error_type / error_stage / tests_passed / tests_total / code_length
+ (policy 추가) policy_action / policy_reason / repeated_same_error

Action 정의:
generate   첫 번째 코드 생성 (initial)
retry      error message 없이 이전 코드 기반 refinement
repair     error message 포함 feedback 기반 수정
plan       planner를 호출해 구현 계획을 생성
stop       더 이상 시도하지 않고 종료
"""

from __future__ import annotations

import gc
import os
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
from src.utils.io import save_result, append_jsonl, make_run_id
from src.utils.dataloader import load_task_and_adapter
from src.utils.prompt_loader import (
    build_planner_prompt_for_sample,
    build_coder_prompt_for_sample,
)
from src.utils.prompting.planner_coder import extract_planner_output


# ─────────────────────────────────────────────
# 1단계 : PolicyState + choose_action
# ─────────────────────────────────────────────

@dataclass
class PolicyState:
    """한 trajectory 안에서 step마다 갱신되는 상태 객체."""

    # 기본 정보
    dataset: str
    call_count: int = 0
    max_calls: int = 3

    # 마지막 step 결과
    exec_ok: bool = False
    test_pass: bool = False
    status: Optional[str] = None
    error_stage: Optional[str] = None   # "EXEC_FAIL" | "TEST_FAIL" | None
    error_type: Optional[str] = None
    tests_passed: int = 0
    tests_total: int = 0
    code_length: int = 0

    # policy 추적
    repeated_same_error: bool = False
    last_action: Optional[str] = None
    last_error_type: Optional[str] = None  # 동일 에러 반복 감지용

    # 계획 존재 여부
    has_plan: bool = False

    # history
    action_history: List[str] = field(default_factory=list)

    def update_from_attempt(self, attempt_record) -> None:
        """attempt_record에서 state 필드를 갱신."""
        new_error_type = attempt_record.error_type

        # 이전 에러와 동일한지 체크
        if (
            self.last_error_type is not None
            and new_error_type is not None
            and self.last_error_type == new_error_type
        ):
            self.repeated_same_error = True
        else:
            self.repeated_same_error = False

        self.exec_ok = attempt_record.exec_ok
        self.test_pass = attempt_record.test_pass
        self.status = attempt_record.status
        self.error_stage = attempt_record.error_stage
        self.error_type = new_error_type
        self.last_error_type = new_error_type
        self.tests_passed = attempt_record.tests_passed
        self.tests_total = attempt_record.tests_total


# ─── action 이름 상수 ────────────────────────
ACTION_GENERATE = "generate"
ACTION_RETRY    = "retry"
ACTION_REPAIR   = "repair"
ACTION_PLAN     = "plan"
ACTION_STOP     = "stop"


def choose_action(state: PolicyState) -> tuple[str, str]:
    """
    현재 PolicyState를 보고 (action, reason) 을 반환.

    우선순위:
      1. 아직 한 번도 생성하지 않았으면 → generate (또는 plan)
      2. 이미 통과했으면 → stop
      3. 예산 소진 → stop
      4. 실행 자체 실패(EXEC_FAIL) + 같은 에러 반복 + 계획 없음 → plan
      5. 실행 자체 실패(EXEC_FAIL) → repair (error message 활용)
      6. 테스트 실패(TEST_FAIL) + 반복 에러 → repair
      7. 테스트 실패(TEST_FAIL) → retry (pure refinement)
      8. 나머지 → repair
    """

    # (1) 첫 호출 - 아직 어떤 action도 없음
    if state.call_count == 0:
        if state.has_plan:
            # plan이 이미 있으면 바로 generate (planner+repair 시나리오)
            return ACTION_GENERATE, "첫 번째 호출 (plan 보유): 코드 생성 시작"
        else:
            return ACTION_GENERATE, "첫 번째 호출: 코드 생성 시작"

    # (2) 이미 통과
    if state.test_pass:
        return ACTION_STOP, "테스트 통과 → 종료"

    # (3) 예산 소진
    if state.call_count >= state.max_calls:
        return ACTION_STOP, f"예산 소진 ({state.call_count}/{state.max_calls}) → 종료"

    # (4) EXEC_FAIL + 반복 에러 + 계획 없음 → 계획부터 세우기
    if (
        state.error_stage == "EXEC_FAIL"
        and state.repeated_same_error
        and not state.has_plan
        # plan은 budget 1회 소모하므로, 남은 budget이 2 이상일 때만
        and (state.max_calls - state.call_count) >= 2
    ):
        return ACTION_PLAN, "EXEC_FAIL 반복 + 계획 없음 → planner 호출로 구조 재설계"

    # (5) EXEC_FAIL → repair (에러 메시지 활용)
    if state.error_stage == "EXEC_FAIL":
        return ACTION_REPAIR, "실행 실패(EXEC_FAIL) → error message 기반 repair"

    # (6) TEST_FAIL + 반복 에러 → repair (logic fix 필요)
    if state.error_stage == "TEST_FAIL" and state.repeated_same_error:
        return ACTION_REPAIR, "TEST_FAIL 반복 → repair로 logic 수정"

    # (7) TEST_FAIL → retry (pure refinement)
    if state.error_stage == "TEST_FAIL":
        return ACTION_RETRY, "테스트 실패(TEST_FAIL) → error 없이 refinement (retry)"

    # (8) 기본 fallback
    return ACTION_REPAIR, "알 수 없는 실패 → 기본 repair"


# ─────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────

def _gen(model: HFModel, prompt: str, max_tokens: Optional[int] = None) -> dict:
    """HFModel.generate() 래퍼 — max_tokens 일시 override."""
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
    """
    code-producing step 공통 실행 함수.
    반환: gen_result / raw_text / generated_code / exec_result / attempt_record / latency
    """
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
    # ── policy 로그 (3단계) ──
    policy_action: Optional[str] = None,
    policy_reason: Optional[str] = None,
    policy_state_snapshot: Optional[dict] = None,
) -> dict:
    """step-level log 엔트리 생성 (policy 필드 포함)."""
    is_planner = stage == "plan"
    is_repair  = stage == "repair"
    is_retry   = stage == "retry"

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
        "exec_ok":       None if is_planner else attempt_record.exec_ok,
        "test_pass":     None if is_planner else attempt_record.test_pass,
        "status":        "PLAN_DONE" if is_planner else attempt_record.status,
        "error_type":    None if is_planner else attempt_record.error_type,
        "error_stage":   None if is_planner else attempt_record.error_stage,
        "error_message": None if is_planner else attempt_record.error_message,
        "tests_passed":  None if is_planner else attempt_record.tests_passed,
        "tests_total":   None if is_planner else attempt_record.tests_total,
        "code_length":   0 if is_planner else (len(code) if code else 0),
        "selected": None,
        "selection_rank": None,
        # 3단계: policy 로그
        "policy_action": policy_action,
        "policy_reason": policy_reason,
        "policy_state":  policy_state_snapshot,
    }

    if planner_text is not None:
        entry["plan_text"] = planner_text

    if hasattr(sample, "entry_point"):
        entry["entry_point"] = sample.entry_point

    return entry


# ─────────────────────────────────────────────
# 2단계 : unified runner
# ─────────────────────────────────────────────

def run_adaptive_policy(config_path: str):
    """
    Adaptive Policy Unified Runner.

    기존의 repair / retry / planner_coder_repair를 별도로 실행하는 구조 대신,
    choose_action()이 매 step마다 최적 action을 선택하여 하나의 루프에서 처리.
    """

    # ── 1. Config 로드 ──────────────────────────
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cfg     = config.get("run", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg   = config.get("model", {})
    method_cfg  = config.get("method", {})
    budget_cfg  = config.get("budget", {})
    output_cfg  = config.get("output", {})
    logging_cfg = config.get("logging", {})
    policy_cfg  = config.get("policy", {})  # adaptive policy 전용 설정

    run_id       = make_run_id(config)
    seed         = run_cfg.get("seed", 42)
    dataset_name = dataset_cfg.get("name", "humaneval")
    num_samples  = dataset_cfg.get("num_samples", 1)

    model_name      = model_cfg.get("name", "")
    max_new_tokens  = model_cfg.get("max_new_tokens", 512)
    temperature     = model_cfg.get("temperature", 0.0)

    # planner 설정 (선택적)
    planner_cfg         = model_cfg.get("planner", {})
    planner_name        = planner_cfg.get("name", model_name)
    planner_tokens      = planner_cfg.get("max_new_tokens", 256)
    planner_temperature = planner_cfg.get("temperature", temperature)

    method_name  = method_cfg.get("name", "adaptive_policy")
    max_calls    = budget_cfg.get("max_calls", 4)

    # policy 옵션
    allow_plan   = policy_cfg.get("allow_plan", True)   # planner 사용 여부
    use_planner_extract = policy_cfg.get("use_planner_extract", False)

    output_dir = output_cfg.get(
        "dir", f"results/phase1_ver3/{dataset_name}/adaptive_policy"
    )

    save_step_level       = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary  = logging_cfg.get("save_problem_summary", True)
    save_run_analysis     = logging_cfg.get("save_run_analysis", True)
    save_code             = logging_cfg.get("save_code", True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🧭 Adaptive Policy 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"max_calls           : {max_calls}")
    print(f"allow_plan          : {allow_plan}")
    print(f"model               : {model_name}")
    print(f"max_new_tokens      : {max_new_tokens}")
    print(f"temperature         : {temperature}")
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
            "policy": policy_cfg,
            "config_path": config_path,
        },
        os.path.join(output_dir, "config.json"),
    )

    # ── 2. Task / Adapter 로드 ──────────────────
    task, adapter = load_task_and_adapter(dataset_name)
    print(f"📦 데이터셋: {dataset_name} | size={len(task)}")

    # ── 3. 모델 로드 ────────────────────────────
    print(f"🔄 모델 로딩: {model_name}")
    main_model = HFModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    print("✅ 모델 로딩 완료")

    # planner 모델 (동일 모델이면 공유)
    if allow_plan:
        if planner_name == model_name:
            planner_model = main_model
            print("✅ Planner = main model (공유)")
        else:
            print(f"🔄 Planner 모델 로딩: {planner_name}")
            planner_model = HFModel(
                model_name=planner_name,
                max_new_tokens=planner_tokens,
                temperature=planner_temperature,
            )
            print("✅ Planner 모델 로딩 완료")
    else:
        planner_model = None

    # ── 4. 실험 루프 ────────────────────────────
    step_log_path       = os.path.join(output_dir, "step_logs.jsonl")
    trajectory_log_path = os.path.join(output_dir, "trajectory_logs.jsonl")

    if save_step_level and os.path.exists(step_log_path):
        os.remove(step_log_path)
    if save_trajectory_level and os.path.exists(trajectory_log_path):
        os.remove(trajectory_log_path)

    eval_results        = []
    written_steps       = 0
    written_trajectories = 0
    transition_counts   = {}
    failure_type_counts = {}
    sum_tokens  = 0.0
    sum_latency = 0.0
    sum_calls   = 0

    samples_to_run  = min(num_samples, len(task))
    global_step_id  = 0

    for i in range(samples_to_run):
        sample       = task.get_sample(i)
        problem_id   = sample.task_id
        trajectory_id = f"{problem_id}_run0"

        print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

        # 문제별 누적 추적
        cumulative_total_tokens = 0
        cumulative_latency      = 0.0
        transition_path         = []
        num_exec_fail           = 0
        num_test_fail           = 0
        call_count              = 0

        # 컨텍스트 보관
        previous_code   : Optional[str] = None
        error_message   : Optional[str] = None
        planner_output  : Optional[str] = None
        last_attempt    : Optional[dict] = None
        final_exec_result              = None

        # ── PolicyState 초기화 ──
        state = PolicyState(
            dataset   = dataset_name,
            call_count = 0,
            max_calls  = max_calls,
        )

        # ── 단일 루프 (unified runner) ──
        while True:

            # 3단계: policy 로그용 state snapshot (action 결정 시점)
            state_snapshot = {
                "call_count":          state.call_count,
                "exec_ok":             state.exec_ok,
                "test_pass":           state.test_pass,
                "status":              state.status,
                "error_stage":         state.error_stage,
                "error_type":          state.error_type,
                "tests_passed":        state.tests_passed,
                "tests_total":         state.tests_total,
                "code_length":         state.code_length,
                "repeated_same_error": state.repeated_same_error,
                "last_action":         state.last_action,
                "has_plan":            state.has_plan,
            }

            # ── 1단계: action 선택 ──
            action, reason = choose_action(state)

            # plan이 비활성화된 경우 plan → repair 로 대체
            if action == ACTION_PLAN and not allow_plan:
                action = ACTION_REPAIR
                reason += " (plan 비활성화 → repair로 대체)"

            print(f"  📌 action={action} | {reason}")

            # ── stop 처리 ──
            if action == ACTION_STOP:
                print(f"  ⏹  STOP")
                break

            # ── plan 처리 ──
            if action == ACTION_PLAN:
                planner_prompt = build_planner_prompt_for_sample(sample)

                t0 = time.perf_counter()
                plan_gen = _gen(planner_model, planner_prompt, planner_tokens)
                plan_latency = time.perf_counter() - t0

                planner_output = extract_planner_output(plan_gen["text"])
                state.has_plan = True

                call_count += 1
                cumulative_total_tokens += plan_gen["total_tokens"]
                cumulative_latency      += plan_latency
                state.call_count         = call_count
                transition_path.append("PLAN_DONE")

                plan_entry = _make_step_entry(
                    run_id          = run_id,
                    dataset_name    = dataset_name,
                    problem_id      = problem_id,
                    method_name     = method_name,
                    trajectory_id   = trajectory_id,
                    step_id         = global_step_id,
                    call_index      = call_count - 1,
                    stage           = "plan",
                    gen_result      = plan_gen,
                    latency_sec     = plan_latency,
                    planner_text    = planner_output,
                    save_code       = save_code,
                    sample          = sample,
                    policy_action   = action,
                    policy_reason   = reason,
                    policy_state_snapshot = state_snapshot,
                )

                if save_step_level:
                    append_jsonl(plan_entry, step_log_path)
                    written_steps += 1

                global_step_id += 1
                state.last_action = ACTION_PLAN
                state.action_history.append(ACTION_PLAN)

                print(f"  📝 Plan: {planner_output[:80].replace(chr(10), ' ')}...")

                # plan 이후 바로 다음 loop iteration에서 generate 실행
                continue

            # ── generate / retry / repair prompt 구성 ──
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
                # 이 분기에 도달하면 안 됨
                raise ValueError(f"Unknown action: {action}")

            # ── 모델 호출 + 평가 ──
            attempt = _run_codegen(
                sample         = sample,
                model          = main_model,
                adapter        = adapter,
                prompt         = prompt,
                method_name    = method_name,
                model_name     = model_name,
                attempt_idx    = call_count,
                use_planner_extract = use_planner_extract and state.has_plan,
                max_tokens     = None,
            )

            call_count += 1
            cumulative_total_tokens += attempt["gen_result"]["total_tokens"]
            cumulative_latency      += attempt["latency"]
            state.call_count         = call_count

            # state 갱신
            state.update_from_attempt(attempt["attempt_record"])
            state.code_length  = len(attempt["generated_code"]) if attempt["generated_code"] else 0
            state.last_action  = action
            state.action_history.append(action)

            current_status = attempt["attempt_record"].status
            transition_path.append(current_status)

            if str(current_status).startswith("EXEC_FAIL"):
                num_exec_fail += 1
            if str(current_status).startswith("TEST_FAIL"):
                num_test_fail += 1

            # 컨텍스트 업데이트
            previous_code  = attempt["generated_code"]
            error_message  = attempt["attempt_record"].error_message
            last_attempt   = attempt
            final_exec_result = attempt["exec_result"]

            # 3단계: step log 기록 (policy 필드 포함)
            step_entry = _make_step_entry(
                run_id          = run_id,
                dataset_name    = dataset_name,
                problem_id      = problem_id,
                method_name     = method_name,
                trajectory_id   = trajectory_id,
                step_id         = global_step_id,
                call_index      = call_count - 1,
                stage           = stage,
                gen_result      = attempt["gen_result"],
                latency_sec     = attempt["latency"],
                code            = attempt["generated_code"],
                attempt_record  = attempt["attempt_record"],
                save_code       = save_code,
                sample          = sample,
                policy_action   = action,
                policy_reason   = reason,
                policy_state_snapshot = state_snapshot,
            )

            if save_step_level:
                append_jsonl(step_entry, step_log_path)
                written_steps += 1

            global_step_id += 1

            pretty = "✅ PASS" if current_status == "PASS" else f"❌ {current_status}"
            print(f"  {stage} (call {call_count - 1}): {pretty}")

            del step_entry

        # ── 문제 종료 ──────────────────────────────
        # last_attempt가 None이면 (action=stop이 첫 action인 비정상 케이스) 처리
        if last_attempt is None:
            print(f"  ⚠️  action이 즉시 STOP → 결과 없음.")
            continue

        eval_results.append(final_exec_result)

        # trajectory log
        final_attempt_record = last_attempt["attempt_record"]
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
            "action_history":     state.action_history,   # policy 전용
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
            failure_type_counts[final_status] = (
                failure_type_counts.get(final_status, 0) + 1
            )

        # OOM 방지
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del trajectory_entry, last_attempt, final_exec_result
        del state
        gc.collect()

    # ── 5. 결과 요약 ───────────────────────────
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
    print("📌 Failure Breakdown")
    print(f"  code_failed:      {extra_summary['code_failed']}")
    print(f"  setup_failed:     {extra_summary['setup_failed']}")
    print(f"  test_failed:      {extra_summary['test_failed']}")
    print(f"  semantic_failed:  {extra_summary['semantic_failed']}")
    print(f"  execution_failed: {extra_summary['execution_failed']}")
    print(f"{'=' * 60}")

    # ── 6. Problem-level summary ────────────────
    n = len(eval_results)
    avg_tokens  = sum_tokens  / n if n else 0.0
    avg_latency = sum_latency / n if n else 0.0
    avg_calls   = sum_calls   / n if n else 0.0

    problem_summary = {
        "run_id":                 run_id,
        "dataset":                dataset_name,
        "method":                 method_name,
        "total_problems":         summary["total"],
        "num_pass":               summary["passed"],
        "pass_at_1":              summary["pass@1"],
        "execution_success_rate": summary["execution_success_rate"],
        "conditional_pass":       summary["conditional_pass"],
        "avg_tokens":             avg_tokens,
        "avg_latency":            avg_latency,
        "avg_calls":              avg_calls,
        "extra_summary":          extra_summary,
    }

    # ── 7. run-level analysis ───────────────────
    run_analysis = {
        "run_id":              run_id,
        "dataset":             dataset_name,
        "method":              method_name,
        "transition_counts":   transition_counts,
        "failure_type_counts": failure_type_counts,
    }

    # ── 8. 결과 저장 ────────────────────────────
    if save_step_level:
        print(f"💾 step_logs    : {step_log_path} ({written_steps}건)")
    if save_trajectory_level:
        print(f"💾 trajectory   : {trajectory_log_path} ({written_trajectories}건)")

    if save_problem_summary:
        save_result(problem_summary, os.path.join(output_dir, "summary.json"))

    if save_run_analysis:
        save_result(run_analysis, os.path.join(output_dir, "analysis.json"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.adaptive_policy <config.yaml>")
        sys.exit(1)

    run_adaptive_policy(sys.argv[1])