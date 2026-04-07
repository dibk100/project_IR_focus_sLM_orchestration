"""
메트릭 계산
HumanEval 평가 지표: pass@k
"""
from typing import List
from src.evaluation.executor import ExecutionResult


def pass_at_1(results: List[ExecutionResult]) -> float:
    """pass@1 계산: 전체 문제 중 통과한 비율"""
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.passed)
    return passed / len(results)


def summarize_results(results: List[ExecutionResult]) -> dict:
    """실행 결과 요약 통계"""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.timeout)
    timed_out = sum(1 for r in results if r.timeout)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "timed_out": timed_out,
        "pass@1": pass_at_1(results),
    }
