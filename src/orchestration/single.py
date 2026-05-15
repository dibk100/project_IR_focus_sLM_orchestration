# src/orchestration/single.py
import gc
import json
import os
import sys
import time
import yaml
from types import SimpleNamespace


from src.models.hf_model_vllm import HFModel

from src.evaluation.metrics import summarize_failure_breakdown, summarize_phase1_results
from src.utils.io import save_result, save_results_jsonl, make_run_id
from src.utils.dataloader import load_task_and_adapter

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0")


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


def _print_sample_execution_flow(sample, prompt: str, raw_text: str, generated_code: str):
    _print_header("SAMPLE DEBUG :: STEP 1. INPUT PROMPT")
    print(_shorten(prompt))

    _print_header("SAMPLE DEBUG :: STEP 2. RAW MODEL OUTPUT (y)")
    print(_shorten(raw_text))

    _print_header("SAMPLE DEBUG :: STEP 3. EXECUTABLE CODE")
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

    elif hasattr(sample, "test"):
        _print_header("SAMPLE DEBUG :: TEST CODE")
        print(_shorten(sample.test))


def _append_jsonl(path: str, record: dict):
    """단일 레코드를 jsonl 파일에 즉시 append"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_completed_ids(step_log_path: str) -> set:
    """기존 step_logs.jsonl에서 완료된 problem_id 목록 복원"""
    completed = set()
    if not os.path.exists(step_log_path):
        return completed
    with open(step_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add(row["problem_id"])
            except Exception:
                continue
    return completed


def run_single_shot(config_path: str):
    """Single-shot baseline 실험 실행"""
    # 1. Config 로드
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

    method_name = method_cfg.get("name", "single_shot")
    max_calls = budget_cfg.get("max_calls", 1)

    output_dir = output_cfg.get("dir", f"results/RUN/{dataset_name}/single")

    save_step_level = logging_cfg.get("save_step_level", True)
    save_trajectory_level = logging_cfg.get("save_trajectory_level", True)
    save_problem_summary = logging_cfg.get("save_problem_summary", True)
    save_run_analysis = logging_cfg.get("save_run_analysis", True)
    save_code = logging_cfg.get("save_code", True)

    debug_mode = debug_cfg.get("mode", "run")

    os.makedirs(output_dir, exist_ok=True)

    # 체크포인트 경로 미리 정의
    step_log_path = os.path.join(output_dir, "step_logs.jsonl")
    trajectory_log_path = os.path.join(output_dir, "trajectory_logs.jsonl")

    # sample 모드일 때 print를 파일로 저장
    stdout_backup = sys.stdout
    log_fp = None
    if debug_mode == "sample":
        debug_log_path = os.path.join(output_dir, "sample_debug_output.txt")
        log_fp = open(debug_log_path, "w", encoding="utf-8")
        sys.stdout = log_fp

    try:
        print("=" * 60)
        print("📋 Single-Shot Baseline 실험")
        print("=" * 60)
        print(f"run_id              : {run_id}")
        print(f"dataset             : {dataset_name}")
        print(f"method              : {method_name}")
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

        # 2. Task / Adapter 로드
        task, adapter = load_task_and_adapter(dataset_name)
        print(f"📦 데이터셋: {dataset_name} | size={len(task)}")

        # ── CHECKPOINT: 이미 완료된 problem_id 복원 ──────────────────
        completed_ids = _load_completed_ids(step_log_path)
        if completed_ids:
            print(f"⏭️  체크포인트 감지: {len(completed_ids)}개 문제 스킵")
        # ─────────────────────────────────────────────────────────────

        # 3. 모델 로드
        print(f"🔄 모델 로딩: {model_name}")
        model = HFModel(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            backend=model_cfg.get("backend", "hf"),
            api_base=model_cfg.get("api_base", None),
        )
        print("✅ 모델 로딩 완료")

        # 4. 실험 실행
        # 요약 통계용 in-memory 리스트 (재시작 후에도 resume된 결과 포함해야 하므로 완료분 재로드)
        step_logs = []
        trajectory_logs = []
        eval_results = []

        # resume 시 기존 결과를 메모리에 복원 (summary 계산용)
        if completed_ids:
            if os.path.exists(step_log_path):
                with open(step_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            step_logs.append(json.loads(line))
            if os.path.exists(trajectory_log_path):
                with open(trajectory_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            traj = json.loads(line)
                            trajectory_logs.append(traj)
                            # eval_results 복원 (passed 여부만 필요)
                            eval_results.append(SimpleNamespace(
                                status=traj["final_status"],
                                passed=(traj["final_status"] == "PASS"),
                                exec_ok=(traj["final_status"] not in ("TOKEN_OVERFLOW",)),
                                test_pass=(traj["final_status"] == "PASS"),
                                tests_passed=traj.get("final_tests_passed", 0),
                                tests_total=traj.get("final_tests_total", 0),
                                error_type=None,
                                error_stage=None,
                                num_calls=traj.get("call_count", 1),
                            ))

        samples_to_run = min(num_samples, len(task))

        for i in range(samples_to_run):
            sample = task.get_sample(i)
            problem_id = sample.task_id

            # ── CHECKPOINT: 완료된 문제 스킵 ─────────────────────────
            if problem_id in completed_ids:
                print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} --- ⏭️ SKIP")
                continue
            # ─────────────────────────────────────────────────────────

            trajectory_id = f"{problem_id}_run0"

            print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

            # 4-1. prompt 구성
            prompt = adapter.build_initial_prompt(sample)

            # 4-2. 모델 호출 + 시간 측정
            gen_start = time.perf_counter()
            try:
                gen_result = model.generate(prompt)
            except Exception as e:
                gen_end = time.perf_counter()
                print(f"  ⚠️ 모델 호출 실패 (토큰 초과 등), 스킵: {e}")
                step_entry = {
                    "run_id": run_id, "dataset": dataset_name,
                    "problem_id": problem_id, "method": method_name,
                    "trajectory_id": trajectory_id,
                    "step_id": 0, "call_index": 0, "candidate_id": 0,
                    "stage": "generate", "is_retry": False, "is_repair": False, "is_planner": False,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "latency_sec": gen_end - gen_start,
                    "code": None, "exec_ok": False, "test_pass": False,
                    "status": "TOKEN_OVERFLOW", "error_type": "TokenOverflow",
                    "error_stage": "generate", "error_message": str(e)[:500],
                    "tests_passed": 0, "tests_total": 0,
                    "code_length": 0, "selected": None, "selection_rank": None,
                }
                if hasattr(sample, "entry_point"):
                    step_entry["entry_point"] = sample.entry_point

                trajectory_entry = {
                    "run_id": run_id, "dataset": dataset_name,
                    "problem_id": problem_id, "method": method_name,
                    "trajectory_id": trajectory_id,
                    "num_steps": 1, "call_count": 1,
                    "final_status": "TOKEN_OVERFLOW", "failure_family": "TOKEN_OVERFLOW",
                    "final_tests_passed": 0, "final_tests_total": 0,
                    "total_tokens": 0, "total_latency": gen_end - gen_start,
                    "transition_path": ["TOKEN_OVERFLOW"],
                    "budget_used": {"tokens": 0, "calls": 1, "latency": gen_end - gen_start},
                }

                step_logs.append(step_entry)
                trajectory_logs.append(trajectory_entry)

                # ── CHECKPOINT: 즉시 디스크에 flush ──────────────────
                if save_step_level:
                    _append_jsonl(step_log_path, step_entry)
                if save_trajectory_level:
                    _append_jsonl(trajectory_log_path, trajectory_entry)
                # ─────────────────────────────────────────────────────

                eval_results.append(SimpleNamespace(
                    status="TOKEN_OVERFLOW", passed=False,
                    exec_ok=False, test_pass=False,
                    tests_passed=0, tests_total=0,
                    error_type="TokenOverflow", error_stage="generate",
                    num_calls=0,
                ))
                gc.collect()
                continue

            gen_end = time.perf_counter()
            latency_sec = gen_end - gen_start

            raw_text = gen_result["text"]
            input_tokens = gen_result["input_tokens"]
            output_tokens = gen_result["output_tokens"]
            total_tokens = gen_result["total_tokens"]

            # 4-3. 코드 추출
            generated_code = adapter.extract_code(sample, raw_text)

            if debug_mode == "sample":
                _print_sample_execution_flow(
                    sample=sample,
                    prompt=prompt,
                    raw_text=raw_text,
                    generated_code=generated_code,
                )

            # 4-4. 실행 / 평가
            exec_result = adapter.execute(sample, generated_code)
            eval_results.append(exec_result)

            if debug_mode == "sample":
                _print_header("SAMPLE DEBUG :: EXEC RESULT OBJECT")
                print(exec_result)

                _print_header("SAMPLE DEBUG :: CLASSIFIED EXECUTION")
                print(adapter.classify_execution(exec_result))

            # 4-5. attempt record 생성
            attempt_record = adapter.make_attempt_record(
                sample=sample,
                method=method_name,
                model_name=model_name,
                attempt_idx=0,
                prompt=prompt,
                raw_output=raw_text,
                generated_code=generated_code,
                latency_sec=latency_sec,
                exec_result=exec_result,
            )

            if debug_mode == "sample":
                _print_header("SAMPLE DEBUG :: ATTEMPT RECORD")
                print(attempt_record)

            # 4-6. step-level log
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
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_sec": latency_sec,
                "code": generated_code if save_code else None,
                "exec_ok": attempt_record.exec_ok,
                "test_pass": attempt_record.test_pass,
                "status": attempt_record.status,
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

            # 4-7. trajectory-level log
            final_status = attempt_record.status

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
                "num_steps": 1,
                "call_count": 1,
                "final_status": final_status,
                "failure_family": failure_family,
                "final_tests_passed": attempt_record.tests_passed,
                "final_tests_total": attempt_record.tests_total,
                "total_tokens": total_tokens,
                "total_latency": latency_sec,
                "transition_path": [final_status],
                "budget_used": {
                    "tokens": total_tokens,
                    "calls": 1,
                    "latency": latency_sec,
                },
            }

            step_logs.append(step_entry)
            trajectory_logs.append(trajectory_entry)

            # ── CHECKPOINT: 즉시 디스크에 flush ──────────────────────
            if save_step_level:
                _append_jsonl(step_log_path, step_entry)
            if save_trajectory_level:
                _append_jsonl(trajectory_log_path, trajectory_entry)
            # ─────────────────────────────────────────────────────────

            pretty_status = (
                "✅ PASS"
                if final_status == "PASS"
                else f"❌ {final_status}"
            )
            print(f"  {pretty_status}")

            del gen_result, step_entry, trajectory_entry
            del attempt_record, exec_result
            gc.collect()

        # 5. 결과 요약
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
        print(f"{'=' * 60}")

        extra_summary = {}
        extra_summary = summarize_failure_breakdown(eval_results)
        print(f"  code_failed: {extra_summary['code_failed']}")
        print(f"  define_test_failed: {extra_summary['define_test_failed']}")
        print(f"  run_test_failed: {extra_summary['run_test_failed']}")
        print(f"{'=' * 60}")

        # 6. Problem-level summary
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

        success_key = f"success@{max_calls}"

        problem_summary = {
            "run_id": run_id,
            "dataset": dataset_name,
            "method": method_name,
            "max_calls": max_calls,
            "total_problems": summary["total"],
            "num_success": summary["success"],
            "success_metric_name": success_key,
            "success_at_k": summary[success_key],
            "execution_success_rate": summary["execution_success_rate"],
            "conditional_success": summary["conditional_success"],
            "avg_tokens": avg_tokens,
            "avg_latency": avg_latency,
            "avg_calls": avg_calls,
            "extra_summary": extra_summary,
        }

        # 7. Run-level analysis summary
        transition_counts = {}
        failure_type_counts = {}

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

        failure_family_counts = {}

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

        # 8. 결과 저장 — step/trajectory는 이미 flush됐으므로 summary/analysis만 저장
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

    finally:
        if log_fp is not None:
            log_fp.close()
            sys.stdout = stdout_backup
            print(f"sample debug output saved to: {debug_log_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.single <config.yaml>")
        sys.exit(1)

    run_single_shot(sys.argv[1])
