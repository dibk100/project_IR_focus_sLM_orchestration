# src/evaluation/executor.py
"""
코드 실행기
- _exec_with_timeout: exec() 무한 루프 방지 (signal.alarm, 10초)
- _cleanup_namespace: exec() 후 namespace + sys.modules 정리 (OOM 방지)
- 모든 executor 함수는 try/finally로 cleanup 보장
"""
import io
import sys
import gc
import signal
import traceback
import contextlib
from dataclasses import dataclass
from typing import Optional, List
import unittest

# ── 타임아웃 설정 (초) ──
EXEC_TIMEOUT_SEC = 30


class ExecutionTimeout(Exception):
    """exec() 실행 타임아웃"""
    pass


def _timeout_handler(signum, frame):
    raise ExecutionTimeout(f"Code execution timed out after {EXEC_TIMEOUT_SEC}s")


def _exec_with_timeout(code_str: str, namespace: dict, timeout: int = EXEC_TIMEOUT_SEC):
    """exec()를 타임아웃과 함께 실행. 무한 루프 방지."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        exec(code_str, namespace)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _cleanup_namespace(namespace: dict, modules_before: set):
    """exec()에서 생성된 namespace와 새로 import된 모듈을 정리"""
    namespace.clear()
    new_modules = set(sys.modules.keys()) - modules_before
    for mod_name in new_modules:
        try:
            del sys.modules[mod_name]
        except KeyError:
            pass
    gc.collect()


@dataclass
class HumanEvalExecutionTrace:
    """HumanEval용 단계별 실행 추적 결과"""
    code_exec_passed: bool
    setup_exec_passed: bool   # exec(test) 성공 여부
    test_exec_passed: bool    # check(candidate) 성공 여부
    passed: bool
    timeout: bool = False

    failed_stage: Optional[str] = None      # "code" / "define_test" / "run_test"
    error_type: Optional[str] = None
    error: Optional[str] = None
    output: str = ""

@dataclass
class MBPPExecutionTrace:
    """MBPP용 단계별 실행 추적 결과"""
    code_exec_passed: bool
    setup_exec_passed: bool
    test_exec_passed: bool
    passed: bool

    failed_stage: Optional[str] = None      # "code" / "define_test" / "run_test"
    failed_test_index: Optional[int] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    output: str = ""


@dataclass
class BigCodeExecutionTrace:
    """BigCode용 단계별 실행 추적 결과"""
    code_exec_passed: bool
    setup_exec_passed: bool
    test_exec_passed: bool
    passed: bool
    timeout: bool = False

    failed_stage: Optional[str] = None      # "code" / "define_test" / "run_test"
    failed_test_index: Optional[int] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    output: str = ""


def _extract_error_type(error_text: str) -> Optional[str]:
    """
    traceback 문자열의 마지막 줄에서 Python 예외 타입을 대략 추출
    예:
        AssertionError
        NameError
        SyntaxError
        TypeError
    """
    if not error_text:
        return None

    last_line = error_text.strip().split("\n")[-1]

    if ":" in last_line:
        return last_line.split(":", 1)[0].strip()

    return last_line.strip()


def execute_humaneval(
    code: str,
    test: str,
    entry_point: str,
) -> HumanEvalExecutionTrace:
    """
    HumanEval 확인 (staged execution)

    실행 단계:
    1. exec(code)               -> candidate 함수 정의
    2. exec(test)               -> check 함수 정의
    3. check(candidate)         -> assert 기반 평가

    failed_stage 정의:
    - "code"         : generated code 자체가 exec 실패
    - "define_test"  : test/check 정의 코드 exec 실패
    - "run_test"     : check(candidate) 실행 실패
    """
    namespace = {}
    stdout_buffer = io.StringIO()
    modules_before = set(sys.modules.keys())

    if not code.strip():
        return HumanEvalExecutionTrace(
            code_exec_passed=False,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",
            error_type="EmptyGeneration",
            error="Empty generation",
            output="",
        )

    try:
        # Stage 1: generated code 실행
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                _exec_with_timeout(code, namespace)
        except Exception:
            err = traceback.format_exc()
            return HumanEvalExecutionTrace(
                code_exec_passed=False, setup_exec_passed=False, test_exec_passed=False,
                passed=False, failed_stage="code",
                error_type=_extract_error_type(err), error=err,
                output=stdout_buffer.getvalue(),
            )

        # entry_point 존재 확인
        if entry_point not in namespace:
            return HumanEvalExecutionTrace(
                code_exec_passed=True, setup_exec_passed=False, test_exec_passed=False,
                passed=False, failed_stage="code",
                error_type="MissingEntryPoint",
                error=f"Entry point '{entry_point}' not found after exec(code)",
                output=stdout_buffer.getvalue(),
            )

        candidate = namespace[entry_point]
        if not callable(candidate):
            return HumanEvalExecutionTrace(
                code_exec_passed=True, setup_exec_passed=False, test_exec_passed=False,
                passed=False, failed_stage="code",
                error_type="NonCallableEntryPoint",
                error=f"Entry point '{entry_point}' exists but is not callable",
                output=stdout_buffer.getvalue(),
            )

        # Stage 2: test code 실행 (check 함수 정의)
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                _exec_with_timeout(test, namespace)
        except Exception:
            err = traceback.format_exc()
            return HumanEvalExecutionTrace(
                code_exec_passed=True, setup_exec_passed=False, test_exec_passed=False,
                passed=False, failed_stage="define_test",
                error_type=_extract_error_type(err), error=err,
                output=stdout_buffer.getvalue(),
            )

        if "check" not in namespace:
            return HumanEvalExecutionTrace(
                code_exec_passed=True, setup_exec_passed=True, test_exec_passed=False,
                passed=False, failed_stage="define_test",
                error_type="MissingCheckFunction",
                error="check function not found after exec(test)",
                output=stdout_buffer.getvalue(),
            )

        check_fn = namespace["check"]
        if not callable(check_fn):
            return HumanEvalExecutionTrace(
                code_exec_passed=True, setup_exec_passed=True, test_exec_passed=False,
                passed=False, failed_stage="define_test",
                error_type="NonCallableCheckFunction",
                error="'check' exists after exec(test) but is not callable",
                output=stdout_buffer.getvalue(),
            )

        # Stage 3: check(candidate) 실행
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                check_fn(candidate)
        except Exception:
            err = traceback.format_exc()
            return HumanEvalExecutionTrace(
                code_exec_passed=True, setup_exec_passed=True, test_exec_passed=False,
                passed=False, failed_stage="run_test",
                error_type=_extract_error_type(err), error=err,
                output=stdout_buffer.getvalue(),
            )

        return HumanEvalExecutionTrace(
            code_exec_passed=True, setup_exec_passed=True, test_exec_passed=True,
            passed=True, output=stdout_buffer.getvalue(),
        )
    finally:
        _cleanup_namespace(namespace, modules_before)


def execute_mbpp_staged(
    code: str,
    test_list: List[str],
    test_setup_code: str = "",
) -> MBPPExecutionTrace:
    """
    MBPP 확인 (staged execution)

    실행 단계:
    1. exec(code)
    2. exec(test_setup_code)   # 있을 경우
    3. for test_case in test_list: exec(test_case)
    
    failed_stage 정의:
    - "code"         : generated code 자체가 exec 실패
    - "define_test"  : test/check 정의 코드 exec 실패
    - "run_test"     : 개별 test assertion 실행 실패

    반환:
    - 어느 단계에서 실패했는지
    - 몇 번째 테스트에서 실패했는지
    - 예외 타입 / traceback
    """
    
    # exec()가 실행될 공유 네임스페이스(dict)
    # ------------------------------------
    # Python의 exec(code, namespace)를 쓰면,
    # code 안에서 정의된 함수/클래스/변수들이 namespace에 저장된다.
    #
    # 예를 들어 code 안에:
    #   class Pair: ...
    #   def max_chain_length(...): ...
    # 가 있으면,
    # namespace 안에 Pair, max_chain_length가 생긴다.
    #
    # 이후 test_setup_code와 test_list도 같은 namespace에서 실행해야
    # 앞에서 정의된 함수/클래스를 그대로 참조할 수 있다.
    namespace = {}
    stdout_buffer = io.StringIO()
    modules_before = set(sys.modules.keys())

    if not code.strip():
        return MBPPExecutionTrace(
            code_exec_passed=False, setup_exec_passed=False, test_exec_passed=False,
            passed=False, failed_stage="code",
            error_type="EmptyGeneration", error="Empty generation", output="",
        )

    try:
        # =========================================================
        # Stage 1: generated code 실행
        # =========================================================
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                _exec_with_timeout(code, namespace)
        except Exception:
            err = traceback.format_exc()
            return MBPPExecutionTrace(
                code_exec_passed=False, setup_exec_passed=False, test_exec_passed=False,
                passed=False, failed_stage="code",
                error_type=_extract_error_type(err), error=err,
                output=stdout_buffer.getvalue(),
            )

        # =========================================================
        # Stage 2: test_setup_code 실행
        # =========================================================
        try:
            if test_setup_code.strip():
                with contextlib.redirect_stdout(stdout_buffer):
                    _exec_with_timeout(test_setup_code, namespace)
        except Exception:
            err = traceback.format_exc()
            return MBPPExecutionTrace(
                code_exec_passed=True, setup_exec_passed=False, test_exec_passed=False,
                passed=False, failed_stage="define_test",
                error_type=_extract_error_type(err), error=err,
                output=stdout_buffer.getvalue(),
            )

        # =========================================================
        # Stage 3: test assertions 실행
        # =========================================================
        try:
            for idx, test_case in enumerate(test_list):
                with contextlib.redirect_stdout(stdout_buffer):
                    _exec_with_timeout(test_case, namespace)
        except Exception:
            err = traceback.format_exc()
            return MBPPExecutionTrace(
                code_exec_passed=True, setup_exec_passed=True, test_exec_passed=False,
                passed=False, failed_stage="run_test",
                failed_test_index=idx,
                error_type=_extract_error_type(err), error=err,
                output=stdout_buffer.getvalue(),
            )

        # =========================================================
        # 모든 단계 성공
        # =========================================================
        return MBPPExecutionTrace(
            code_exec_passed=True, setup_exec_passed=True, test_exec_passed=True,
            passed=True, output=stdout_buffer.getvalue(),
        )
    finally:
        _cleanup_namespace(namespace, modules_before)

    
def execute_bigcode(
    code: str,
    test: str,
) -> BigCodeExecutionTrace:
    """
    BigCode 평가 (staged execution)

    실행 단계:
    1. exec(code)   -> candidate 함수 정의
    2. exec(test)   -> unittest 테스트 정의
    3. unittest 실행 -> 실제 테스트 수행

    failed_stage 정의:
    - "code"         : generated code 자체가 exec 실패
    - "define_test"  : test/unittest 정의 코드 exec 실패
    - "run_test"     : unittest 실행 실패
    """

    namespace = {}
    stdout_buffer = io.StringIO()
    modules_before = set(sys.modules.keys())

    if not code.strip():
        return BigCodeExecutionTrace(
            code_exec_passed=False, setup_exec_passed=False, test_exec_passed=False,
            passed=False, failed_stage="code",
            error_type="EmptyGeneration", error="Empty generation", output="",
        )

    try:
        # =========================================================
        # Stage 1: generated code 실행
        # =========================================================
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                _exec_with_timeout(code, namespace)
        except Exception:
            err = traceback.format_exc()
            return BigCodeExecutionTrace(
                code_exec_passed=False, setup_exec_passed=False, test_exec_passed=False,
                passed=False, failed_stage="code",
                error_type=_extract_error_type(err), error=err,
                output=stdout_buffer.getvalue(),
            )

        # =========================================================
        # Stage 2: test code 실행 (unittest class/function 정의)
        # =========================================================
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                _exec_with_timeout(test, namespace)
        except Exception:
            err = traceback.format_exc()
            return BigCodeExecutionTrace(
                code_exec_passed=True, setup_exec_passed=False, test_exec_passed=False,
                passed=False, failed_stage="define_test",
                error_type=_extract_error_type(err), error=err,
                output=stdout_buffer.getvalue(),
            )

        # =========================================================
        # Stage 2-1: TestCases 확인
        # =========================================================
        if "TestCases" not in namespace:
            return BigCodeExecutionTrace(
                code_exec_passed=True,
                setup_exec_passed=True,
                test_exec_passed=False,
                passed=False,
                failed_stage="define_test",
                error_type="MissingTestCases",
                error="TestCases class not found after exec(test)",
                output=stdout_buffer.getvalue(),
            )

        test_cls = namespace["TestCases"]

        # TestCases가 존재는 하는데 class가 아닌 경우
        if not isinstance(test_cls, type):
            return BigCodeExecutionTrace(
                code_exec_passed=True,
                setup_exec_passed=True,
                test_exec_passed=False,
                passed=False,
                failed_stage="define_test",
                error_type="InvalidTestCases",
                error="'TestCases' exists after exec(test) but is not a class",
                output=stdout_buffer.getvalue(),
            )

        # TestCases가 class이긴 한데 unittest.TestCase를 상속하지 않은 경우
        if not issubclass(test_cls, unittest.TestCase):
            return BigCodeExecutionTrace(
                code_exec_passed=True,
                setup_exec_passed=True,
                test_exec_passed=False,
                passed=False,
                failed_stage="define_test",
                error_type="InvalidTestCases",
                error="'TestCases' exists but is not a unittest.TestCase subclass",
                output=stdout_buffer.getvalue(),
            )

        # =========================================================
        # Stage 3: unittest 실행
        # =========================================================
        try:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_cls)

            if suite.countTestCases() == 0:
                return BigCodeExecutionTrace(
                    code_exec_passed=True,
                    setup_exec_passed=True,
                    test_exec_passed=False,
                    passed=False,
                    failed_stage="define_test",
                    error_type="NoTestCases",
                    error="No test methods found in TestCases",
                    output=stdout_buffer.getvalue(),
                )

            run_stream = io.StringIO()
            runner = unittest.TextTestRunner(stream=run_stream, verbosity=0)

            with contextlib.redirect_stdout(stdout_buffer):
                result = runner.run(suite)

            if not result.wasSuccessful():
                details = run_stream.getvalue().strip()

                error_traces = [tb for _, tb in result.errors]
                failure_traces = [tb for _, tb in result.failures]

                if error_traces:
                    error_type = _extract_error_type(error_traces[0])
                elif failure_traces:
                    error_type = _extract_error_type(failure_traces[0])
                else:
                    error_type = "TestFailure"

                sections = []

                if error_traces:
                    sections.append("[unittest errors]")
                    sections.extend(error_traces)

                if failure_traces:
                    sections.append("[unittest failures]")
                    sections.extend(failure_traces)

                if details:
                    sections.append("[unittest summary]")
                    sections.append(details)

                error_text = "\n\n".join(sections).strip()
                if not error_text:
                    error_text = "unittest reported failure without details"

                return BigCodeExecutionTrace(
                    code_exec_passed=True,
                    setup_exec_passed=True,
                    test_exec_passed=False,
                    passed=False,
                    failed_stage="run_test",
                    error_type=error_type,
                    error=error_text,
                    output=stdout_buffer.getvalue(),
                )

        # unittest runner 자체가 터진 경우
        except Exception:
            err = traceback.format_exc()
            return BigCodeExecutionTrace(
                code_exec_passed=True,
                setup_exec_passed=True,
                test_exec_passed=False,
                passed=False,
                failed_stage="run_test",
                error_type=_extract_error_type(err),
                error=err,
                output=stdout_buffer.getvalue(),
            )

        # =========================================================
        # 모든 단계 성공
        # =========================================================
        return BigCodeExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=True,
            test_exec_passed=True,
            passed=True,
            output=stdout_buffer.getvalue(),
        )
    finally:
        _cleanup_namespace(namespace, modules_before)