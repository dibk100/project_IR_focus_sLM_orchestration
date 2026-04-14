"""
Phase 1 메트릭 계산

- pass@1 (= overall pass rate)
- execution success rate (= structural success)
- conditional pass (= semantic success among executable samples)
- gain vs single
- MBPP failure breakdown (분석용)

1) execution success rate
   - 생성 코드가 '평가 가능한 상태'까지 갔는지, exec(code) 코드실행의 성공 비율
   - structural quality를 반영

2) pass@1
   - 전체 문제 중 최종적으로 정답인 비율
   - overall performance를 반영

3) conditional pass
   - 실행 가능한 코드들 중 실제 정답인 비율
   - exec(code) 문제가 없는 것들 중 실제 정답인 비율
   - semantic quality를 반영

즉,
- orchestration이 execution_success_rate를 올리면 구조 개선
- orchestration이 conditional_pass를 올리면 의미 개선
- pass@1은 전체 효과


"""
from typing import Any, Dict, List, Optional, Union
from collections import Counter

from src.evaluation.executor import ExecutionResult, MBPPExecutionTrace


# HumanEval 결과와 MBPP 결과, 그리고 ver3 dict/AttemptRecord 스타일까지 함께 다룸
ResultType = Union[ExecutionResult, MBPPExecutionTrace, Dict[str, Any], Any]


def _get_attr(result: Any, key: str, default=None):
    """
    dict / dataclass / object 모두에서 안전하게 값을 꺼내는 헬퍼
    """
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def is_passed(result: ResultType) -> bool:
    """
    최종 정답 여부를 반환

    우선순위:
    1) passed 필드
    2) test_pass 필드
    """
    passed = _get_attr(result, "passed", None)
    if passed is not None:
        return bool(passed)

    test_pass = _get_attr(result, "test_pass", None)
    if test_pass is not None:
        return bool(test_pass)

    return False


def is_exec_success(result: ResultType) -> bool:
    """
    execution success 여부를 반환.

    우선순위:
    1) exec_ok
    2) exec_success
    3) status
    4) 구버전 ExecutionResult / MBPPExecutionTrace 규칙
    """
    exec_ok = _get_attr(result, "exec_ok", None)
    if exec_ok is not None:
        return bool(exec_ok)

    exec_success = _get_attr(result, "exec_success", None)
    if exec_success is not None:
        return bool(exec_success)

    status = _get_attr(result, "status", None)
    if isinstance(status, str):
        if status == "PASS":
            return True
        if status.startswith("TEST_FAIL:"):
            return True
        if status.startswith("EXEC_FAIL:"):
            return False

    if isinstance(result, MBPPExecutionTrace):
        return result.code_exec_passed and result.setup_exec_passed

    if isinstance(result, ExecutionResult):
        if result.passed:
            return True
        if result.error_type == "AssertionError":
            return True
        return False

    return False


def pass_at_1(results: List[ResultType]) -> float:
    """
    pass@1 계산
    """
    if not results:
        return 0.0

    passed_count = sum(1 for r in results if is_passed(r))
    return passed_count / len(results)


def execution_success_rate(results: List[ResultType]) -> float:
    """
    execution success rate 계산
    """
    if not results:
        return 0.0

    exec_success_count = sum(1 for r in results if is_exec_success(r))
    return exec_success_count / len(results)


def conditional_pass(results: List[ResultType]) -> float:
    """
    conditional pass 계산
    """
    if not results:
        return 0.0

    exec_success_count = sum(1 for r in results if is_exec_success(r))
    if exec_success_count == 0:
        return 0.0

    passed_count = sum(1 for r in results if is_passed(r))
    return passed_count / exec_success_count


def gain_vs_single(current_pass_at_1: float, single_pass_at_1: float) -> float:
    """
    single-shot baseline 대비 성능 향상량 계산
    """
    return current_pass_at_1 - single_pass_at_1


def summarize_phase1_results(
    results: List[ResultType],
    single_baseline_pass_at_1: Optional[float] = None,
) -> dict:
    """
    Phase 1 핵심 지표를 한 번에 요약
    """
    total = len(results)
    passed_count = sum(1 for r in results if is_passed(r))
    exec_success_count = sum(1 for r in results if is_exec_success(r))

    summary = {
        "total": total,
        "passed": passed_count,
        "exec_success": exec_success_count,
        "pass@1": pass_at_1(results),
        "execution_success_rate": execution_success_rate(results),
        "conditional_pass": conditional_pass(results),
    }

    if single_baseline_pass_at_1 is not None:
        summary["gain_vs_single"] = gain_vs_single(
            summary["pass@1"],
            single_baseline_pass_at_1,
        )

    return summary


def summarize_failure_type_counts(results: List[ResultType]) -> dict:
    """
    ver3 공통 failure type 집계
    - PASS 제외
    - status가 있으면 status 기준
    """
    counter = Counter()

    for r in results:
        status = _get_attr(r, "status", None)
        if isinstance(status, str) and status != "PASS":
            counter[status] += 1

    return dict(counter)


def summarize_transition_counts(trajectory_logs: List[Dict[str, Any]]) -> dict:
    """
    trajectory log 기반 전이 집계
    transition_path의 coarse prefix 기준으로 집계
    예:
      EXEC_FAIL:TypeError -> TEST_FAIL:AssertionError
      => EXEC_FAIL->TEST_FAIL
    """
    counter = Counter()

    for traj in trajectory_logs:
        path = traj.get("transition_path", [])
        for i in range(len(path) - 1):
            a = str(path[i]).split(":")[0]
            b = str(path[i + 1]).split(":")[0]
            counter[f"{a}->{b}"] += 1

    return dict(counter)


def summarize_mbpp_failure_breakdown(results: List[MBPPExecutionTrace]) -> dict:
    """
    MBPP 전용 failure breakdown 요약
    """
    stage_counter = Counter(r.failed_stage for r in results if r.failed_stage)
    error_counter = Counter(r.error_type for r in results if r.error_type)

    code_failed = sum(1 for r in results if r.failed_stage == "code")
    setup_failed = sum(1 for r in results if r.failed_stage == "setup")
    test_failed = sum(1 for r in results if r.failed_stage == "test")

    semantic_failed = sum(
        1
        for r in results
        if r.failed_stage == "test" and r.error_type == "AssertionError"
    )

    execution_failed = sum(
        1
        for r in results
        if r.failed_stage == "test" and r.error_type != "AssertionError"
    )

    return {
        "failed_stage_breakdown": dict(stage_counter),
        "error_type_breakdown": dict(error_counter),
        "code_failed": code_failed,
        "setup_failed": setup_failed,
        "test_failed": test_failed,
        "semantic_failed": semantic_failed,
        "execution_failed": execution_failed,
    }
