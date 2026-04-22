"""
Repair Loop Orchestration

흐름: Problem → G → V → R → Evaluate
1. task 읽기
2. initial generation
3. 실행/평가 (Verification)
4. 실패 시 (입력 + 이전코드 + 에러메시지)로 repair prompt 생성 → 재생성
5. 최대 max_repair 횟수까지 반복

retry와의 차이:
- retry: error message 없이 pure refinement (이전 코드 + 원래 문제만 전달)
- repair: error message를 포함한 feedback 기반 수정 (exec fail, test fail 정보 포함)

핵심:
- execution feedback(에러 메시지)를 활용한 구조적 수정
- 각 loop마다 어떤 단계(EXEC_FAIL, TEST_FAIL)에서 실패했고,
  다음 시도에서 해결되었는지 transition_path로 추적

Phase1 ver3 기준:
- nested config 구조 사용
- HFModel.generate()의 구조화된 반환값 사용
- step_logs / trajectory_logs / summary / analysis 저장
"""
import gc
import io
import os
import sys
import time
import yaml
import contextlib

import torch

from src.models.hf_model import HFModel

from src.evaluation.metrics import summarize_failure_breakdown, summarize_phase1_results
from src.utils.io import save_result, save_results_jsonl, make_run_id
from src.utils.dataloader import load_task_and_adapter


def _shorten(text: str | None, max_len: int = 8000) -> str:
    if text is None:
        return "None"
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n\n... [truncated, total_len={len(text)}]"


def _print_header(title: str, width: int = 80):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def _print_sample_execution_flow(sample, prompt: str, raw_text: str, generated_code: str | None = None):
    """
    sample 모드에서 사람이 눈으로 pipeline 흐름을 따라갈 수 있도록
    단계별 input/output을 출력한다.

    generated_code=None이면 STEP 1(prompt), STEP 2(raw_text)만 출력하고
    STEP 3 이후 실행 단계는 스킵한다. (빈 출력 차단 경로에서 사용)

    특히 HumanEval은 아래 흐름을 그대로 보여준다:
    (1) input: prompt
    (2) output y
    (3) code = prompt + y
    (4) exec(code) -> candidate 생성
    (5) exec(test) -> check 생성
    (6) check(candidate) -> assert 평가
    """
    _print_header("SAMPLE DEBUG :: STEP 1. INPUT PROMPT")
    print(_shorten(prompt))

    _print_header("SAMPLE DEBUG :: STEP 2. RAW MODEL OUTPUT (y)")
    print(_shorten(raw_text))

    _print_header("SAMPLE DEBUG :: STEP 3. EXECUTABLE CODE")
    if generated_code is None:
        print("(empty or not yet extracted — skipping execution steps)")
        return
    print(_shorten(generated_code))

    # HumanEval 전용 흐름 상세 출력
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

    # MBPP
    elif hasattr(sample, "test_list"):
        _print_header("SAMPLE DEBUG :: MBPP TEST SETUP")
        if hasattr(sample, "test_setup_code"):
            print(_shorten(sample.test_setup_code))
        else:
            print("None")

        _print_header("SAMPLE DEBUG :: MBPP TEST LIST")
        for idx, test_case in enumerate(sample.test_list):
            print(f"[test_list[{idx}]]")
            print(_shorten(test_case))
            print("-" * 40)

    # BigCode
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
    """
    실패 유형(status)별 대표 예시 1건만 수집한다.
    이미 동일 status의 예시가 있으면 스킵한다.
    """
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


def run_repair_loop(config_path: str):
    """Repair loop 실험 실행"""
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
    debug_cfg = config.get("debug", {})

    run_id = make_run_id(config)
    seed = run_cfg.get("seed", 42)

    dataset_name = dataset_cfg.get("name", "humaneval")
    num_samples = dataset_cfg.get("num_samples", 1)

    model_name = model_cfg["name"]
    max_new_tokens = model_cfg.get("max_new_tokens", 512)
    temperature = model_cfg.get("temperature", 0.0)

    method_name = method_cfg.get("name", "repair_loop")

    max_calls = budget_cfg.get("max_calls", 3)
    max_repair = budget_cfg.get("max_repair", 2)

    output_dir = output_cfg.get("dir", f"results/RUN/{dataset_name}/repair")

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    # debug mode
    debug_mode = debug_cfg.get("mode", "run")   # "run" | "sample"

    os.makedirs(output_dir, exist_ok=True)

    # sample 모드일 때 print를 파일로 저장
    stdout_backup = sys.stdout
    log_fp = None
    if debug_mode == "sample":
        debug_log_path = os.path.join(output_dir, "sample_debug_output.txt")
        log_fp = open(debug_log_path, "w", encoding="utf-8")
        sys.stdout = log_fp

    try:
        print("=" * 60)
        print("🔧 Repair Loop 실험")
        print("=" * 60)
        print(f"run_id              : {run_id}")
        print(f"dataset             : {dataset_name}")
        print(f"method              : {method_name}")
        print(f"max_calls           : {max_calls}")
        print(f"max_repair          : {max_repair}")
        print(f"max_new_tokens      : {max_new_tokens}")
        print(f"temperature         : {temperature}")
        print(f"seed                : {seed}")
        print(f"output_dir          : {output_dir}")
        print(f"debug_mode          : {debug_mode}")
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
                "debug": debug_cfg,
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

            # 문제별 누적 토큰/레이턴시/상태 추적
            cumulative_input_tokens = 0
            cumulative_output_tokens = 0
            cumulative_total_tokens = 0
            cumulative_latency = 0.0
            transition_path = []
            num_exec_fail = 0
            num_test_fail = 0
            call_count = 0

            previous_code = None
            final_attempt_record = None

            # initial prompt
            current_prompt = adapter.build_initial_prompt(sample)

            for attempt_idx in range(max_repair + 1):
                is_repair = attempt_idx > 0

                # 4-1. 모델 호출 + 시간 측정
                gen_start = time.perf_counter()
                gen_result = model.generate(current_prompt)
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

                # 4-1-check. 모델 출력이 빈 값인지 확인
                if output_tokens == 0 or not raw_text.strip():
                    current_status = "CODE_FAIL:empty_output"
                    transition_path.append(current_status)

                    step_entry = {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "problem_id": problem_id,
                        "method": method_name,
                        "trajectory_id": trajectory_id,
                        "step_id": attempt_idx,
                        "call_index": attempt_idx,
                        "candidate_id": 0,
                        "stage": "generate" if not is_repair else "repair",
                        "is_retry": False,
                        "is_repair": is_repair,
                        "is_planner": False,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "latency_sec": latency_sec,
                        "code": None,
                        "exec_ok": False,
                        "test_pass": False,
                        "status": current_status,
                        "error_type": "empty_output",
                        "error_stage": "code",
                        "error_message": "Model output was empty or contained no extractable code.",
                        "tests_passed": 0,
                        "tests_total": None,
                        "code_length": 0,
                        "selected": None,
                        "selection_rank": None,
                    }

                    if hasattr(sample, "entry_point"):
                        step_entry["entry_point"] = sample.entry_point

                    step_logs.append(step_entry)

                    final_attempt_record = SimpleNamespace(
                        status=current_status,
                        tests_passed=0,
                        tests_total=None,
                        passed=False,
                    )

                    print(f"  attempt {attempt_idx}: ❌ {current_status}")

                    _collect_failure_example(
                        failure_examples,
                        problem_id=problem_id,
                        attempt_idx=attempt_idx,
                        status=current_status,
                        prompt=current_prompt,
                        raw_text=raw_text,
                        generated_code=None,
                        error_type="empty_output",
                        error_stage="code",
                        error_message="Model output was empty or contained no extractable code.",
                    )

                    if attempt_idx == max_repair:
                        break
                    # 다음 repair는 previous_code=None(초기 상태)으로 재시도
                    current_prompt = adapter.build_repair_prompt(
                        sample=sample,
                        previous_code=previous_code,
                        error_message="empty_output: no code was generated.",
                    )
                    continue

                # 4-2. 코드 추출
                generated_code = adapter.extract_code(sample, raw_text)

                # sample 모드일 때 단계별 실행 흐름 출력
                if debug_mode == "sample":
                    _print_sample_execution_flow(
                        sample=sample,
                        prompt=current_prompt,
                        raw_text=raw_text,
                        generated_code=generated_code,
                    )

                # 4-3. 실행 / 평가 (Verification 단계)
                exec_result = adapter.execute(sample, generated_code)

                if debug_mode == "sample":
                    _print_header("SAMPLE DEBUG :: EXEC RESULT OBJECT")
                    print(exec_result)

                    _print_header("SAMPLE DEBUG :: CLASSIFIED EXECUTION")
                    print(adapter.classify_execution(exec_result))

                # 4-4. attempt record 생성
                attempt_record = adapter.make_attempt_record(
                    sample=sample,
                    method=method_name,
                    model_name=model_name,
                    attempt_idx=attempt_idx,
                    prompt=current_prompt,
                    raw_output=raw_text,
                    generated_code=generated_code,
                    latency_sec=latency_sec,
                    exec_result=exec_result,
                )

                if debug_mode == "sample":
                    _print_header("SAMPLE DEBUG :: ATTEMPT RECORD")
                    print(attempt_record)

                final_attempt_record = attempt_record
                current_status = attempt_record.status
                transition_path.append(current_status)

                if str(current_status).startswith("EXEC_FAIL"):
                    num_exec_fail += 1
                if str(current_status).startswith("TEST_FAIL"):
                    num_test_fail += 1

                # 4-5. step-level log
                step_entry = {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "problem_id": problem_id,
                    "method": method_name,
                    "trajectory_id": trajectory_id,
                    "step_id": attempt_idx,
                    "call_index": attempt_idx,
                    "candidate_id": 0,
                    "stage": "generate" if not is_repair else "repair",
                    "is_retry": False,
                    "is_repair": is_repair,
                    "is_planner": False,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "latency_sec": latency_sec,
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
                    step_entry["entry_point"] = sample.entry_point

                step_logs.append(step_entry)

                pretty_status = (
                    "✅ PASS"
                    if current_status == "PASS"
                    else f"❌ {current_status}"
                )
                print(f"  attempt {attempt_idx}: {pretty_status}")

                _collect_failure_example(
                    failure_examples,
                    problem_id=problem_id,
                    attempt_idx=attempt_idx,
                    status=current_status,
                    prompt=current_prompt,
                    raw_text=raw_text,
                    generated_code=generated_code,
                    error_type=attempt_record.error_type,
                    error_stage=attempt_record.error_stage,
                    error_message=attempt_record.error_message,
                )

                # 성공하면 종료
                if attempt_record.passed:
                    break

                # 마지막 attempt이면 종료
                if attempt_idx == max_repair:
                    break

                # 4-6. 다음 repair를 위한 repair prompt 구성
                #   retry와의 핵심 차이: error_message를 포함하여 전달
                previous_code = generated_code
                current_prompt = adapter.build_repair_prompt(
                    sample=sample,
                    previous_code=previous_code,
                    error_message=attempt_record.error_message,
                )

            # (문제 종료) 최종 exec_result를 eval_results에 추가
            eval_results.append(exec_result)

            # 4-7. trajectory-level log
            final_status = final_attempt_record.status

            if final_status == "PASS":
                failure_family = "PASS"
            else:
                failure_family = str(final_status).split(":")[0]

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

            trajectory_logs.append(trajectory_entry)

            # OOM 방지
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del gen_result, step_entry, trajectory_entry
            del attempt_record, exec_result
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
        extra_summary = summarize_failure_breakdown(eval_results)
        print(f"  code_failed: {extra_summary['code_failed']}")
        print(f"  define_test_failed: {extra_summary['define_test_failed']}")
        print(f"  run_test_failed: {extra_summary['run_test_failed']}")
        print(f"{'=' * 60}")

        # ── 6. Problem-level summary ──
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

        # ── 7. Run-level analysis summary ──
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
            if status != "PASS":
                failure_type_counts[status] = failure_type_counts.get(status, 0) + 1

        for step in step_logs:
            status = step["status"]
            family = "PASS" if status == "PASS" else str(status).split(":")[0]
            if family != "PASS":
                failure_family_counts[family] = failure_family_counts.get(family, 0) + 1

        run_analysis = {
            "run_id": run_id,
            "dataset": dataset_name,
            "method": method_name,
            "transition_counts": transition_counts,
            "failure_type_counts": failure_type_counts,
            "failure_family_counts": failure_family_counts,
        }

        # ── 8. 결과 저장 ──
        if save_step_level:
            save_results_jsonl(
                step_logs,
                os.path.join(output_dir, "step_logs.jsonl"),
            )

        if save_trajectory_level:
            save_results_jsonl(
                trajectory_logs,
                os.path.join(output_dir, "trajectory_logs.jsonl"),
            )

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

        # ── 9. 실패 유형별 대표 예시 저장 ──
        if failure_examples:
            save_result(
                failure_examples,
                os.path.join(output_dir, "failure_examples.json"),
            )
            print(f"📝 failure_examples: {len(failure_examples)}개 유형 저장됨")

    finally:
        # sample 모드에서 stdout 복구
        if log_fp is not None:
            log_fp.close()
            sys.stdout = stdout_backup
            print(f"sample debug output saved to: {debug_log_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.repair <config.yaml>")
        sys.exit(1)

    run_repair_loop(sys.argv[1])