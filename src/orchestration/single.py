# src/orchestration/single.py
import gc
import os
import sys
import time
import yaml
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


def _print_sample_execution_flow(sample, prompt: str, raw_text: str, generated_code: str):
    """
    sample 모드에서 사람이 눈으로 pipeline 흐름을 따라갈 수 있도록
    단계별 input/output을 출력한다.

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

        # 3. 모델 로드
        print(f"🔄 모델 로딩: {model_name}")
        model = HFModel(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print("✅ 모델 로딩 완료")

        # 4. 실험 실행
        step_logs = []
        trajectory_logs = []
        eval_results = []

        samples_to_run = min(num_samples, len(task))

        for i in range(samples_to_run):
            sample = task.get_sample(i)
            problem_id = sample.task_id
            trajectory_id = f"{problem_id}_run0"

            print(f"\n--- [{i + 1}/{samples_to_run}] {problem_id} ---")

            # 4-1. prompt 구성
            prompt = adapter.build_initial_prompt(sample)

            # 4-2. 모델 호출 + 시간 측정
            gen_start = time.perf_counter()
            gen_result = model.generate(prompt)
            gen_end = time.perf_counter()
            latency_sec = gen_end - gen_start

            raw_text = gen_result["text"]
            input_tokens = gen_result["input_tokens"]
            output_tokens = gen_result["output_tokens"]
            total_tokens = gen_result["total_tokens"]
            
            # 4-3. 코드 추출
            generated_code = adapter.extract_code(sample, raw_text)

            # sample 모드일 때 단계별 실행 흐름 출력
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

            step_logs.append(step_entry)

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

            trajectory_logs.append(trajectory_entry)

            pretty_status = (
                "✅ PASS"
                if final_status == "PASS"
                else f"❌ {final_status}"
            )
            print(f"  {pretty_status}")

            # OOM 방지
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del gen_result, step_entry, trajectory_entry
            del attempt_record, exec_result
            gc.collect()
        
        # 5. 결과 요약
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

        # 8. 결과 저장
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

    finally:
        # sample 모드에서 stdout 복구
        if log_fp is not None:
            log_fp.close()
            sys.stdout = stdout_backup
            print(f"sample debug output saved to: {debug_log_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.orchestration.single <config.yaml>")
        sys.exit(1)

    run_single_shot(sys.argv[1])