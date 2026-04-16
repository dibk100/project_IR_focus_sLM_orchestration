"""
코드 실행기
생성된 코드 + 테스트 케이스를 실행하여 pass/fail을 판정

- HumanEval: subprocess 기반 end-to-end 실행
- MBPP: staged exec 기반 디버깅/분석용 실행
"""
import io
import os
import traceback
import contextlib
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ExecutionResult:
    """HumanEval용 실행 결과"""
    passed: bool
    output: str
    error: Optional[str] = None
    timeout: bool = False
    error_type: Optional[str] = None
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None


@dataclass
class MBPPExecutionTrace:
    """MBPP용 단계별 실행 추적 결과"""
    code_exec_passed: bool
    setup_exec_passed: bool
    test_exec_passed: bool
    passed: bool

    failed_stage: Optional[str] = None      # "code" / "setup" / "test"
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

    failed_stage: Optional[str] = None      # "code" / "setup" / "test"
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


def _run_python_file(full_code: str, timeout: int = 10) -> ExecutionResult:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return ExecutionResult(
                passed=True,
                output=result.stdout,
                tests_passed=1,
                tests_total=1,
            )

        error_text = result.stderr.strip()
        return ExecutionResult(
            passed=False,
            output=result.stdout,
            error=error_text,
            error_type=_extract_error_type(error_text),
            tests_passed=0,
            tests_total=1,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False,
            output="",
            error="Timeout",
            timeout=True,
            error_type="Timeout",
            tests_passed=0,
            tests_total=1,
        )
    except Exception as e:
        return ExecutionResult(
            passed=False,
            output="",
            error=str(e),
            error_type=type(e).__name__,
            tests_passed=0,
            tests_total=1,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def execute_humaneval(
    code: str,
    test: str,
    entry_point: str,
    timeout: int = 5,
) -> ExecutionResult:
    """
    HumanEval 평가:
    generated code + test code + check(entry_point)
    
    Args:
        code: 완성된 함수 코드
        test: HumanEval 테스트 코드 (check 함수 포함)
        entry_point: 함수 이름
        timeout: 실행 제한 시간 (초)
    
    Returns:
        ExecutionResult
    """
    full_code = code + "\n" + test + f"\ncheck({entry_point})\n"
    return _run_python_file(full_code, timeout=timeout)


def execute_mbpp_staged(
    code: str,
    test_list: List[str],
    test_setup_code: str = "",
) -> MBPPExecutionTrace:
    """
    MBPP 평가 (Phase 1 분석용 staged execution)

    실행 단계:
    1. exec(code)
    2. exec(test_setup_code)   # 있을 경우
    3. for test_case in test_list: exec(test_case)

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
    
    # stdout(출력)을 가로채기 위한 버퍼
    # --------------------------------
    # generated code나 test code 안에 print()가 있을 수 있다.
    # 그 출력이 화면으로 바로 나가는 대신,
    # 문자열 버퍼(stdout_buffer)에 모이게 하려고 사용한다.
    #
    # 나중에 output=stdout_buffer.getvalue()로 저장 가능하다.
    stdout_buffer = io.StringIO()

    # 빈 생성 결과 방어
    # ----------------
    # 모델이 아예 빈 문자열을 냈다면,
    # exec(code)를 하기 전에 즉시 실패로 처리한다.
    # 이 경우는 syntax/runtime 문제가 아니라
    # "아예 코드 생성이 비어 있음"이라는 별도 failure로 보는 게 좋다.
    # =========================================================
    #### Stage 0: LM이 1차 코드 생성 자체를 못한 이슈로 정의한다.
    # =========================================================
    if not code.strip():
        return MBPPExecutionTrace(
            code_exec_passed=False,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",
            error_type="EmptyGeneration",
            error="Empty generation",
            output="",
        )

    # =========================================================
    # Stage 1: generated code 실행
    # =========================================================
    # 여기서는 모델이 생성한 전체 코드를 exec()한다.
    #
    # 기대하는 것:
    # - 함수 정의 성공
    # - 클래스 정의 성공
    # - syntax error 없음
    #
    # 예:
    #   class Pair: ...
    #   def max_chain_length(...): ...
    #
    # 이 단계에서 실패하면 보통:
    # - SyntaxError
    # - IndentationError
    # - NameError (코드 로드 시점에 바로 참조하는 경우)
    # 등이 나온다.
    # =========================================================
    #### 나는 이 단계의 실패를 Structural failuer로 정의하고자 한다. LM모델의 출력이 구조적 이해 부족으로 발생하는 실패.
    #### 이건 fine-tuning하면 나아질까? 아니면 few-shot?
    # =========================================================
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            # exec(code, namespace):
            # ----------------------
            # code 문자열을 실제 Python 코드처럼 실행한다.
            #
            # 여기서 함수/클래스 정의가 namespace에 등록된다.
            # 중요한 점은, 아래 setup/test도 같은 namespace를 공유한다는 것.
            exec(code, namespace)
        
        # 여기까지 왔다는 건 Stage 1 성공
        code_exec_passed = True
        
    except Exception:
        # traceback.format_exc():
        # -----------------------
        # 방금 발생한 예외의 전체 traceback 문자열을 가져온다.
        # 단순히 예외 타입만 보는 게 아니라,
        # 나중에 디버깅할 수 있도록 전체 stack trace를 저장한다.
        err = traceback.format_exc()
        return MBPPExecutionTrace(
            code_exec_passed=False,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",                  # 실패 단계: code
            error_type=_extract_error_type(err), # 예외 타입 추출 (예: SyntaxError)
            error=err,                           # 전체 traceback 저장
            output=stdout_buffer.getvalue(),     # 지금까지 capture된 stdout
        )
        
    # =========================================================
    # Stage 2: test_setup_code 실행
    # =========================================================
    # MBPP 샘플에 따라 test_setup_code가 비어 있을 수도 있고,
    # import/helper/check 함수 등이 들어 있을 수도 있다.
    #
    # 이 setup도 같은 namespace에서 실행해야
    # generated code에서 만든 함수/클래스와 연결된다.
    try:
        if test_setup_code.strip():
            with contextlib.redirect_stdout(stdout_buffer):
                exec(test_setup_code, namespace)
        setup_exec_passed = True
    except Exception:
        err = traceback.format_exc()
        # code는 성공했지만 setup에서 실패한 경우
        return MBPPExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="setup",
            error_type=_extract_error_type(err),
            error=err,
            output=stdout_buffer.getvalue(),
        )

    # =========================================================
    # Stage 3: test assertions 실행
    # =========================================================
    # test_list의 각 assert를 하나씩 실행한다.
    #
    # 예:
    #   assert max_chain_length(...) == 3
    #   assert max_chain_length(...) == 4
    #
    # 이렇게 하나씩 exec()하면,
    # 어느 번째 테스트에서 실패했는지(idx) 알 수 있다.
    try:
        for idx, test_case in enumerate(test_list):
            with contextlib.redirect_stdout(stdout_buffer):
                exec(test_case, namespace)
                
        # 모든 테스트를 통과하면 성공
        test_exec_passed = True
    except Exception:
        err = traceback.format_exc()
        # code, setup은 성공했지만 test 중 하나에서 실패
        return MBPPExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=True,
            test_exec_passed=False,
            passed=False,
            failed_stage="test",                 # 실패 단계: test
            failed_test_index=idx,               # 몇 번째 테스트에서 실패했는지
            error_type=_extract_error_type(err), # AssertionError / NameError / TypeError 등
            error=err,
            output=stdout_buffer.getvalue(),
        )
    # =========================================================
    # 모든 단계 성공
    # =========================================================
    return MBPPExecutionTrace(
        code_exec_passed=True,
        setup_exec_passed=True,
        test_exec_passed=True,
        passed=True,
        output=stdout_buffer.getvalue(),
    )
    
def execute_bigcode(
    code: str,
    test: str,
    timeout: int = 5,
) -> ExecutionResult:
    """
    실행 단계:
    1. exec(code)
    2. exec(test)
    """
    namespace = {}
    stdout_buffer = io.StringIO()

    if not code.strip():
        return BigCodeExecutionTrace(
            code_exec_passed=False,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",
            error_type="EmptyGeneration",
            error="Empty generation",
            output="",
        )

    # =========================================================
    # Stage 1: generated code 실행
    
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, namespace)

        code_exec_passed = True
        
    except Exception:

        err = traceback.format_exc()
        return BigCodeExecutionTrace(
            code_exec_passed=False,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",                  # 실패 단계: code
            error_type=_extract_error_type(err), # 예외 타입 추출 (예: SyntaxError)
            error=err,                           # 전체 traceback 저장
            output=stdout_buffer.getvalue(),     # 지금까지 capture된 stdout
        )
        
    # =========================================================
    # Stage 2: test실행
    # =========================================================
    
    try:
        if test.strip():
            with contextlib.redirect_stdout(stdout_buffer):
                exec(test, namespace)
        test_passed = True
        
    except Exception:
        err = traceback.format_exc()
        return BigCodeExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=True,
            test_exec_passed=False,
            passed=False,
            failed_stage="test",                 # 실패 단계: test
            error_type=_extract_error_type(err), # AssertionError / NameError / TypeError 등
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