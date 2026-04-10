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
from typing import List, Union, Optional
from collections import Counter

from src.evaluation.executor import ExecutionResult, MBPPExecutionTrace


# HumanEval 결과와 MBPP 결과를 함께 다룰 수 있도록 Union 타입 정의
ResultType = Union[ExecutionResult, MBPPExecutionTrace]


def is_passed(result: ResultType) -> bool:
    """
    최종 정답 여부를 반환

    HumanEval:
    - ExecutionResult.passed 사용

    MBPP:
    - MBPPExecutionTrace.passed 사용
    """
    return result.passed


def is_exec_success(result: ResultType) -> bool:
    """
    execution success 여부를 반환.

    이 함수가 핵심.
    왜냐하면 Phase 1에서 execution success rate와 conditional pass를 계산할 때
    '어디까지를 structural success로 볼 것인가'를 정의하기 때문

    [MBPP]
    - code_exec_passed and setup_exec_passed 를 execution success로 정의함. exec()의 범위를 setup까지 포함하기로 함.
    - 즉, 생성 코드가 로드되고(setup 전까지 통과),
      테스트를 실제로 돌릴 수 있는 상태에 도달해야 함
    - test 단계에서 AssertionError가 나더라도 execution success로 보기로 함.
      (semantic failure는 execution success 이후 발생)

    [HumanEval]
    - 현재 executor가 staged execution이 아니므로 완전히 동일하게 분해되진 않음
    - 따라서 practical definition을 사용:
        1) passed=True 이면 execution success
        2) AssertionError 로 실패한 경우도 execution success
           (즉, 실행은 되었고 정답 assertion에서 실패한 것)
    - 그 외 SyntaxError, NameError, Timeout 등은 execution failure로 본다

    """
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

    문제당 1개 생성한다고 가정하면,
    pass@1은 곧 '최종 정답 비율'과 같다.

    수식:
        pass@1 = (# passed) / (# total)
        
    results는 problems별 faill, pass(실패, 성공) 으로 구성된 리스트
    """
    if not results:
        return 0.0

    passed_count = sum(1 for r in results if is_passed(r))
    return passed_count / len(results)


def execution_success_rate(results: List[ResultType]) -> float:
    """
    execution success rate 계산

    의미:
    - 생성 결과가 최소한 실행 가능한 평가 상태까지 갔는가?

    수식:
        execution_success_rate = (# exec_success) / (# total)

    해석:
    - structural quality 지표
    - syntax, completeness, helper definition, setup readiness 등을 반영
    """
    if not results:
        return 0.0

    exec_success_count = sum(1 for r in results if is_exec_success(r))
    return exec_success_count / len(results)


def conditional_pass(results: List[ResultType]) -> float:
    """
    conditional pass 계산

    의미:
    - 실행 가능한 코드들만 놓고 봤을 때, 실제로 정답인 비율

    수식:
        conditional_pass = (# passed) / (# exec_success)

    해석:
    - semantic quality 지표
    - 구조적으로 실행 가능한 샘플들 중 reasoning / logic correctness를 반영

    예:
    - execution_success_rate는 높은데 conditional_pass가 낮다
      -> 실행은 되지만 논리가 자주 틀림
    - conditional_pass가 높다
      -> 일단 실행 가능한 코드가 나오면 의미적으로는 잘 맞춤
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

    수식:
        gain_vs_single = current_pass@1 - single_pass@1
    """
    return current_pass_at_1 - single_pass_at_1


def summarize_phase1_results(
    results: List[ResultType],
    single_baseline_pass_at_1: Optional[float] = None,
) -> dict:
    """
    Phase 1 핵심 지표를 한 번에 요약.

    포함 지표:
    - total
    - passed
    - exec_success
    - pass@1
    - execution_success_rate
    - conditional_pass
    - gain_vs_single (optional)

    이 함수는 HumanEval / MBPP 공통으로 사용 가능.
    단, failure breakdown은 데이터셋별로 따로 보는 편이 좋다.
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

    # single-shot baseline pass@1이 주어졌다면 gain도 함께 계산
    if single_baseline_pass_at_1 is not None:
        summary["gain_vs_single"] = gain_vs_single(
            summary["pass@1"],
            single_baseline_pass_at_1,
        )

    return summary


def summarize_mbpp_failure_breakdown(results: List[MBPPExecutionTrace]) -> dict:
    """
    MBPP 전용 failure breakdown 요약

    MBPP는 staged execution 결과를 가지고 있으므로,
    실패를 더 자세히 분석할 수 있다.

    breakdown 예시:
    - code stage failure      -> structural failure
    - setup stage failure     -> setup/environment failure
    - test stage + AssertionError
                              -> semantic failure
    - test stage + other error
                              -> execution failure during testing

    반환 항목:
    - failed_stage_breakdown
    - error_type_breakdown
    - code_failed
    - setup_failed
    - test_failed
    - semantic_failed
    - execution_failed
    """
    stage_counter = Counter(r.failed_stage for r in results if r.failed_stage)
    error_counter = Counter(r.error_type for r in results if r.error_type)

    code_failed = sum(1 for r in results if r.failed_stage == "code")
    setup_failed = sum(1 for r in results if r.failed_stage == "setup")
    test_failed = sum(1 for r in results if r.failed_stage == "test")

    # semantic failure:
    # - test 단계까지 갔고
    # - assertion에서 실패한 경우
    semantic_failed = sum(
        1
        for r in results
        if r.failed_stage == "test" and r.error_type == "AssertionError"
    )

    # execution failure:
    # - test 단계에서 실패했지만
    # - assertion failure가 아닌 경우
    # 예: NameError, TypeError 등
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
    
# NOTE:
# Current implementation assumes one result per problem (pass@1 setting).
# If pass@k is introduced later, results should be grouped by problem first.