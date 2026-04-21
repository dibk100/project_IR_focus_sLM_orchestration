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
# src/evaluation/metrics.py
from typing import Any, Dict, List, Optional, Union
from collections import Counter

from src.evaluation.executor import (
    HumanEvalExecutionTrace,
    MBPPExecutionTrace,
    BigCodeExecutionTrace,
)

ResultType = Union[
    HumanEvalExecutionTrace,
    MBPPExecutionTrace,
    BigCodeExecutionTrace,
]


def _get_attr(result: Any, key: str, default=None):
    """
    공통 속성 접근 헬퍼.

    역할:
    - result가 dict이면 result[key]처럼 접근
    - result가 dataclass/object이면 getattr(result, key)로 접근
    - 속성이 없으면 default 반환

    목적:
    - metrics 함수들이 dict / dataclass / object 형태를 가리지 않고
      공통 인터페이스처럼 다룰 수 있게 해준다.
    """
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def is_passed(result: ResultType) -> bool:
    """
    '최종적으로 정답을 맞췄는지'를 판정한다.

    의미:
    - PASS 여부(semantic correctness)를 bool로 변환하는 함수
    - 최종 pass@1 계산의 기본 단위

    판정 우선순위:
    1) passed 필드
    2) test_pass 필드

    반환:
    - True  : 최종 정답
    - False : 오답 또는 실행 실패
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
    '실행 가능한 상태까지 도달했는지'를 판정한다.

    의미:
    - structural success 여부를 bool로 반환
    - 즉, code 실행 + test 정의까지 성공했는지 판정
    - test를 통과했는지와는 별개다

    해석:
    - PASS                  -> execution success
    - TEST_FAIL:*           -> execution success
      (실행은 됐지만 테스트에서 실패)
    - EXEC_FAIL:*           -> execution failure
    - DEFINE_TEST_FAIL:*    -> execution failure
    - OTHER_FAIL:*          -> execution failure

    판정 우선순위:
    1) exec_ok
    2) exec_success
    3) status 문자열
    4) trace 객체 규칙(code_exec_passed and setup_exec_passed)

    반환:
    - True  : 실행 가능한 상태까지 도달
    - False : 실행 자체 실패
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
        if status.startswith("DEFINE_TEST_FAIL:"):
            return False
        if status.startswith("OTHER_FAIL:"):
            return False

    if isinstance(result, HumanEvalExecutionTrace):
        return result.code_exec_passed and result.setup_exec_passed

    if isinstance(result, MBPPExecutionTrace):
        return result.code_exec_passed and result.setup_exec_passed

    if isinstance(result, BigCodeExecutionTrace):
        return result.code_exec_passed and result.setup_exec_passed

    return False


def pass_at_1(results: List[ResultType]) -> float:
    """
    pass@1 계산.

    의미:
    - 전체 문제 중 최종적으로 정답을 맞춘 비율
    - 가장 기본적인 overall 성능 지표

    예:
    - 100문제 중 27문제 PASS -> pass@1 = 0.27
    """
    if not results:
        return 0.0
    passed_count = sum(1 for r in results if is_passed(r))
    return passed_count / len(results)


def execution_success_rate(results: List[ResultType]) -> float:
    """
    execution success rate 계산.

    의미:
    - 전체 문제 중 '실행 가능한 상태'까지 간 비율
    - structural quality를 보는 지표
    - 코드 자체가 깨지지 않고, 테스트를 정의할 수 있었는지를 반영

    예:
    - 100문제 중 60문제가 exec_ok=True -> 0.60
    """
    if not results:
        return 0.0
    exec_success_count = sum(1 for r in results if is_exec_success(r))
    return exec_success_count / len(results)


def conditional_pass(results: List[ResultType]) -> float:
    """
    conditional pass 계산.

    의미:
    - 실행 가능한 코드들 중 실제로 정답인 비율
    - structural failure를 제외하고 semantic quality만 보고 싶을 때 쓰는 지표

    해석:
    conditional_pass = passed / exec_success

    예:
    - 100문제 중 exec_success 50개
    - 그 중 PASS 20개
    -> conditional_pass = 20 / 50 = 0.4
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
    single-shot baseline 대비 성능 향상량 계산.

    의미:
    - 현재 방법의 pass@1이 baseline보다 얼마나 나아졌는지 측정

    예:
    - current = 0.31
    - single  = 0.24
    -> gain_vs_single = 0.07
    """
    return current_pass_at_1 - single_pass_at_1


def summarize_phase1_results(
    results: List[ResultType],
    single_baseline_pass_at_1: Optional[float] = None,
) -> dict:
    """
    Phase 1 핵심 성능 지표를 한 번에 요약한다.

    포함 항목:
    - total                    : 총 문제 수
    - passed                   : 최종 PASS 개수
    - exec_success             : execution success 개수
    - pass@1                   : overall 성능
    - execution_success_rate   : structural quality
    - conditional_pass         : semantic quality
    - gain_vs_single           : baseline 대비 성능 향상량 (옵션)

    역할:
    - 실험 전체 결과를 한 번에 summary dict로 묶어서
      logging / json 저장 / console 출력에 쓰기 좋게 만든다.
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
    status 기준 failure type 빈도를 집계한다.

    의미:
    - PASS를 제외한 status 문자열의 등장 횟수를 센다.
    - 예: EXEC_FAIL:SyntaxError, TEST_FAIL:AssertionError 등

    역할:
    - failure category 분포를 거칠게 보고 싶을 때 사용
    - 어떤 종류의 실패가 많이 발생했는지 빠르게 확인 가능
    """
    counter = Counter()

    for r in results:
        status = _get_attr(r, "status", None)
        if isinstance(status, str) and status != "PASS":
            counter[status] += 1

    return dict(counter)


def summarize_transition_counts(trajectory_logs: List[Dict[str, Any]]) -> dict:
    """
    trajectory log 기반 상태 전이(transition) 빈도를 집계한다.

    의미:
    - transition_path 안의 연속된 상태쌍을 세어서
      coarse 상태 전이 패턴을 요약한다.
    - prefix만 사용하므로 세부 에러 타입은 무시하고
      EXEC_FAIL / TEST_FAIL / PASS 같은 coarse 상태만 본다.

    예:
    - EXEC_FAIL:TypeError -> TEST_FAIL:AssertionError
      => EXEC_FAIL->TEST_FAIL 로 집계

    역할:
    - retry / repair / planner 실험에서
      상태가 어떻게 이동하는지 분석할 때 유용
    """
    counter = Counter()

    for traj in trajectory_logs:
        path = traj.get("transition_path", [])
        for i in range(len(path) - 1):
            a = str(path[i]).split(":")[0]
            b = str(path[i + 1]).split(":")[0]
            counter[f"{a}->{b}"] += 1

    return dict(counter)


def summarize_failure_breakdown(results: List[ResultType]) -> dict:
    """
    trace 기반 failure breakdown 요약.

    stage 정의:
    - "code"         : generated code 자체가 깨짐
    - "define_test"  : 테스트 정의 단계에서 실패
    - "run_test"     : 실제 테스트 실행 단계에서 실패

    포함 항목:
    - failed_stage_breakdown : stage별 실패 빈도
    - error_type_breakdown   : error_type별 실패 빈도
    - code_failed            : code 단계 실패 수
    - define_test_failed     : define_test 단계 실패 수
    - run_test_failed        : run_test 단계 실패 수

    역할:
    - failure를 stage 축으로 보고 싶을 때 사용하는 분석 함수
    - structural failure가 많은지, test 실행까지 갔는지 등을 확인할 수 있음
    """

    def _failed_stage(r) -> Optional[str]:
        return _get_attr(r, "failed_stage", None)

    stage_counter = Counter(
        _failed_stage(r) for r in results if _failed_stage(r) is not None
    )
    error_counter = Counter(
        _get_attr(r, "error_type", None)
        for r in results
        if _get_attr(r, "error_type", None) is not None
    )

    code_failed = sum(1 for r in results if _failed_stage(r) == "code")
    define_test_failed = sum(1 for r in results if _failed_stage(r) == "define_test")
    run_test_failed = sum(1 for r in results if _failed_stage(r) == "run_test")

    return {
        "failed_stage_breakdown": dict(stage_counter),
        "error_type_breakdown": dict(error_counter),
        "code_failed": code_failed,
        "define_test_failed": define_test_failed,
        "run_test_failed": run_test_failed,
    }