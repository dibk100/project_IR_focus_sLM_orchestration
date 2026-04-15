"""
Best-of-N Orchestration

흐름:
Problem → G × N → S → Evaluate

1. task 읽기
2. 동일 prompt로 N개 후보를 독립 생성
3. 각 후보를 실행/평가
4. Selector로 최종 후보 선택
5. 결과 저장

Selector 규칙 (pass_seeking):
- 실행 가능한 후보 우선 선택 → TEST 실행
- 통과한 후보가 있으면 → tie_break(shortest_code)로 최종 선택
- 모두 실패 → fail

핵심:
- 비구조적 다중 시도 방식
- 후보 다양성에 의한 pass-seeking 전략
- retry와 비용(call 수) 동일 budget 하에서 비교 대상

Phase1 ver3 기준:
- nested config 구조 사용
- HFModel.generate()의 구조화된 반환값 사용
- step_logs / trajectory_logs / summary / analysis 저장
- step_logs에 candidate_id, selected, selection_rank 기록
"""
import os
import time
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.models.hf_model import HFModel

from src.tasks.humaneval import HumanEvalTask
from src.tasks.mbpp import MBPPTask

from src.adapters.humaneval import HumanEvalAdapter
from src.adapters.mbpp import MBPPAdapter
from src.adapters.base import AttemptRecord

from src.evaluation.metrics import (
    summarize_phase1_results,
    summarize_mbpp_failure_breakdown,
)

import torch
from src.utils.io import save_result, save_results_jsonl, append_jsonl


def load_task_and_adapter(dataset_name: str):
    """
    dataset 이름에 따라 task loader와 adapter를 함께 반환
    """
    if dataset_name == "humaneval":
        return HumanEvalTask(), HumanEvalAdapter()

    if dataset_name == "mbpp":
        return MBPPTask(), MBPPAdapter()

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_run_id(config: dict) -> str:
    """
    run_id 자동 생성/보정
    - config에 값이 있으면 기본값으로 사용
    - 뒤에 timestamp를 붙여 고유하게 만듦
    """
    base_run_id = config.get("run", {}).get("run_id", "phase1_bestofn")
    suffix = datetime.now().strftime("%m%d%H%M%S")
    return f"{base_run_id}_{suffix}"


# ──────────────────────────────────────────────
#  Selector: N개 후보 중 최종 1개 선택
# ──────────────────────────────────────────────
def select_best_candidate(
    candidates: List[Dict[str, Any]],
    selector_cfg: Dict[str, Any],
) -> int:
    """
    N개 후보의 attempt_record 결과를 받아 최종 선택 index를 반환한다.

    selector_cfg 예시:
        rule: "pass_seeking"
        tie_break: "shortest_code"
        if_no_pass: "fail"

    선택 로직 (pass_seeking):
    1. PASS인 후보가 있으면 → tie_break 규칙으로 선택
    2. PASS는 없지만 exec_ok인 후보가 있으면 → tie_break 규칙으로 선택
    3. 모두 EXEC_FAIL → 첫 번째 후보 반환 (if_no_pass=fail)

    Returns:
        선택된 후보의 index (0-based)
    """
    rule = selector_cfg.get("rule", "pass_seeking")
    tie_break = selector_cfg.get("tie_break", "shortest_code")

    if rule != "pass_seeking":
        # 현재 pass_seeking만 지원, 추후 확장 가능
        raise ValueError(f"Unsupported selector rule: {rule}")

    # 1단계: PASS 후보 필터
    passed_indices = [
        idx for idx, c in enumerate(candidates)
        if c["attempt_record"].passed
    ]

    if passed_indices:
        return _apply_tie_break(candidates, passed_indices, tie_break)

    # 2단계: exec_ok 후보 필터 (실행은 됐지만 테스트 실패)
    exec_ok_indices = [
        idx for idx, c in enumerate(candidates)
        if c["attempt_record"].exec_ok
    ]

    if exec_ok_indices:
        return _apply_tie_break(candidates, exec_ok_indices, tie_break)

    # 3단계: 모두 실패 → 첫 번째 반환
    return 0


def _apply_tie_break(
    candidates: List[Dict[str, Any]],
    indices: List[int],
    tie_break: str,
) -> int:
    """
    tie_break 규칙에 따라 후보 중 하나를 선택한다.

    지원 규칙:
    - "shortest_code": 코드 길이가 가장 짧은 후보
    - "first": 첫 번째 후보
    """
    if len(indices) == 1:
        return indices[0]

    if tie_break == "shortest_code":
        best_idx = indices[0]
        best_len = len(candidates[best_idx]["generated_code"] or "")
        for idx in indices[1:]:
            code_len = len(candidates[idx]["generated_code"] or "")
            if code_len < best_len:
                best_len = code_len
                best_idx = idx
        return best_idx

    # default: first
    return indices[0]


def run_best_of_n(config_path: str):
    """Best-of-N 실험 실행"""
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

    method_name = method_cfg.get("name", "best_of_n")
    num_candidates = method_cfg.get("num_candidates", 3)
    selector_cfg = method_cfg.get("selector", {
        "rule": "pass_seeking",
        "tie_break": "shortest_code",
        "if_no_pass": "fail",
    })

    max_calls = budget_cfg.get("max_calls", 3)

    output_dir = output_cfg.get("dir", f"results/phase1_ver3/{dataset_name}/bestofn")

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🎯 Best-of-N 실험")
    print("=" * 60)
    print(f"run_id              : {run_id}")
    print(f"dataset             : {dataset_name}")
    print(f"method              : {method_name}")
    print(f"num_candidates (N)  : {num_candidates}")
    print(f"max_calls           : {max_calls}")
    print(f"selector            : {selector_cfg}")
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
    step_log_path = os.path.join(output_dir, "step_logs.jsonl")
    trajectory_log_path = os.path.join(output_dir, "trajectory_logs.jsonl")

    if save_step_level and os.path.exists(step_log_path):
        os.remove(step_log_path)
    if save_trajectory_level and os.path.exists(trajectory_log_path):
        os.remove(trajectory_log_path)

    eval_results = []           # 최종 선택된 후보의 exec_result (summarize용)
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

        # 문제별 누적 추적
        cumulative_total_tokens = 0
        cumulative_latency = 0.0
        transition_path = []
        num_exec_fail = 0
        num_test_fail = 0

        # 동일 prompt로 N개 후보 독립 생성
        prompt = adapter.build_initial_prompt(sample)
        candidates = []         # 각 후보 정보 저장

        for cand_idx in range(num_candidates):
            # 4-1. 모델 호출 + 시간 측정
            gen_start = time.perf_counter()
            gen_result = model.generate(prompt)
            gen_end = time.perf_counter()
            latency_sec = gen_end - gen_start

            raw_text = gen_result["text"]
            input_tokens = gen_result["input_tokens"]
            output_tokens = gen_result["output_tokens"]
            total_tokens = gen_result["total_tokens"]

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
                attempt_idx=cand_idx,
                prompt=prompt,
                raw_output=raw_text,
                generated_code=generated_code,
                latency_sec=latency_sec,
                exec_result=exec_result,
            )

            current_status = attempt_record.status
            transition_path.append(current_status)

            if str(current_status).startswith("EXEC_FAIL"):
                num_exec_fail += 1
            if str(current_status).startswith("TEST_FAIL"):
                num_test_fail += 1

            candidates.append({
                "cand_idx": cand_idx,
                "attempt_record": attempt_record,
                "exec_result": exec_result,
                "generated_code": generated_code,
                "raw_text": raw_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_sec": latency_sec,
            })

            pretty_status = (
                "✅ PASS"
                if current_status == "PASS"
                else f"❌ {current_status}"
            )
            print(f"  candidate {cand_idx}: {pretty_status}")

        # ── 4-5. Selector: 최종 후보 선택 ──
        selected_idx = select_best_candidate(candidates, selector_cfg)
        selected = candidates[selected_idx]

        print(f"  → selected: candidate {selected_idx} ({selected['attempt_record'].status})")

        # 최종 선택된 후보의 exec_result를 eval에 사용
        eval_results.append(selected["exec_result"])

        # ── 4-6. step-level log (각 후보별) ──
        for cand_idx, cand in enumerate(candidates):
            ar = cand["attempt_record"]
            is_selected = (cand_idx == selected_idx)

            # selection_rank 계산: 선택된 후보 = 1
            selection_rank = 1 if is_selected else None

            step_entry = {
                "run_id": run_id,
                "dataset": dataset_name,
                "problem_id": problem_id,
                "method": method_name,
                "trajectory_id": trajectory_id,
                "step_id": cand_idx,
                "call_index": cand_idx,
                "candidate_id": cand_idx,
                "stage": "generate",
                "is_retry": False,
                "is_repair": False,
                "is_planner": False,
                "input_tokens": cand["input_tokens"],
                "output_tokens": cand["output_tokens"],
                "total_tokens": cand["total_tokens"],
                "latency_sec": cand["latency_sec"],
                "code": cand["generated_code"] if save_code else None,
                "exec_ok": ar.exec_ok,
                "test_pass": ar.test_pass,
                "status": ar.status,
                "error_type": ar.error_type,
                "error_stage": ar.error_stage,
                "error_message": ar.error_message,
                "tests_passed": ar.tests_passed,
                "tests_total": ar.tests_total,
                "code_length": len(cand["generated_code"]) if cand["generated_code"] is not None else 0,
                "selected": is_selected,
                "selection_rank": selection_rank,
            }

            if hasattr(sample, "entry_point"):
                step_entry["entry_point"] = sample.entry_point

            if save_step_level:
                append_jsonl(step_entry, step_log_path)
                written_steps += 1

        # ── 4-7. trajectory-level log ──
        final_record = selected["attempt_record"]
        # transition_path에 선택 결과도 추가 (SELECT→최종상태)
        transition_path.append(f"SELECT:{final_record.status}")

        trajectory_entry = {
            "run_id": run_id,
            "dataset": dataset_name,
            "problem_id": problem_id,
            "method": method_name,
            "trajectory_id": trajectory_id,
            "num_steps": num_candidates,
            "call_count": num_candidates,
            "final_status": final_record.status,
            "final_tests_passed": final_record.tests_passed,
            "final_tests_total": final_record.tests_total,
            "selected_candidate": selected_idx,
            "total_tokens": cumulative_total_tokens,
            "total_latency": cumulative_latency,
            "num_exec_fail": num_exec_fail,
            "num_test_fail": num_test_fail,
            "transition_path": transition_path,
            "budget_used": {
                "tokens": cumulative_total_tokens,
                "calls": num_candidates,
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

        final_status = trajectory_entry["final_status"]
        if final_status != "PASS":
            failure_type_counts[final_status] = failure_type_counts.get(final_status, 0) + 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    if dataset_name == "mbpp":
        extra_summary = summarize_mbpp_failure_breakdown(eval_results)
        print("📌 MBPP Failure Breakdown")
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
        print("Usage: python -m src.orchestration.best_of_n <config.yaml>")
        sys.exit(1)

    run_best_of_n(sys.argv[1])
