# src/evaluation/executor.py
"""
코드 실행기
"""
import io
import traceback
import contextlib
from dataclasses import dataclass
from typing import Optional, List
import unittest

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

    # timeout 인자는 현재 시그니처 호환용으로 유지
    # (in-process exec에서는 실제 강제 timeout 제어는 하지 않음)

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

    # =========================================================
    # Stage 1: generated code 실행
    # =========================================================
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, namespace)
    except Exception:
        err = traceback.format_exc()
        return HumanEvalExecutionTrace(
            code_exec_passed=False,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",
            error_type=_extract_error_type(err),
            error=err,
            output=stdout_buffer.getvalue(),
        )

    # entry_point 존재 확인
    if entry_point not in namespace:
        return HumanEvalExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",
            error_type="MissingEntryPoint",
            error=f"Entry point '{entry_point}' not found after exec(code)",
            output=stdout_buffer.getvalue(),
        )

    candidate = namespace[entry_point]

    if not callable(candidate):
        return HumanEvalExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",
            error_type="NonCallableEntryPoint",
            error=f"Entry point '{entry_point}' exists but is not callable",
            output=stdout_buffer.getvalue(),
        )

    # =========================================================
    # Stage 2: test code 실행 (check 함수 정의)
    # =========================================================
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(test, namespace)
    except Exception:
        err = traceback.format_exc()
        return HumanEvalExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="define_test",
            error_type=_extract_error_type(err),
            error=err,
            output=stdout_buffer.getvalue(),
        )

    if "check" not in namespace:
        return HumanEvalExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=True,
            test_exec_passed=False,
            passed=False,
            failed_stage="define_test",
            error_type="MissingCheckFunction",
            error="check function not found after exec(test)",
            output=stdout_buffer.getvalue(),
        )

    check_fn = namespace["check"]

    if not callable(check_fn):
        return HumanEvalExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=True,
            test_exec_passed=False,
            passed=False,
            failed_stage="define_test",
            error_type="NonCallableCheckFunction",
            error="'check' exists after exec(test) but is not callable",
            output=stdout_buffer.getvalue(),
        )

    # =========================================================
    # Stage 3: check(candidate) 실행
    # =========================================================
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            check_fn(candidate)
    except Exception:
        err = traceback.format_exc()
        return HumanEvalExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=True,
            test_exec_passed=False,
            passed=False,
            failed_stage="run_test",
            error_type=_extract_error_type(err),
            error=err,
            output=stdout_buffer.getvalue(),
        )

    return HumanEvalExecutionTrace(
        code_exec_passed=True,
        setup_exec_passed=True,
        test_exec_passed=True,
        passed=True,
        output=stdout_buffer.getvalue(),
    )

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
            failed_stage="define_test",
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
            failed_stage="run_test",                 # 실패 단계: test
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

    주의:
    - timeout 인자는 현재 시그니처 호환용으로만 유지한다.
    - in-process exec / unittest 실행에서는 강제 timeout을 구현하지 않는다.
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
    # =========================================================
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, namespace)
    except Exception:
        err = traceback.format_exc()
        return BigCodeExecutionTrace(
            code_exec_passed=False,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="code",
            error_type=_extract_error_type(err),
            error=err,
            output=stdout_buffer.getvalue(),
        )

    # =========================================================
    # Stage 2: test code 실행 (unittest class/function 정의)
    # =========================================================
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(test, namespace)
    except Exception:
        err = traceback.format_exc()
        return BigCodeExecutionTrace(
            code_exec_passed=True,
            setup_exec_passed=False,
            test_exec_passed=False,
            passed=False,
            failed_stage="define_test",
            error_type=_extract_error_type(err),
            error=err,
            output=stdout_buffer.getvalue(),
        )

    # =========================================================
    # Stage 2-1: TestCases 확인
    # =========================================================
    if "TestCases" not in namespace:
        return BigCodeExecutionTrace(
            code_exec_passed=True,
            # BigCode에서는 별도 setup 단계가 없지만,
            # 통합 trace 스키마를 맞추기 위해
            # test definition 단계 성공 여부를 setup_exec_passed에 대응시킨다.
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
    # 함수나 변수로 덮어쓴 경우 -> unittest로 실행 불가능
    if not isinstance(test_cls, type):
        return BigCodeExecutionTrace(
            code_exec_passed=True,
            # setup_exec_passed=True:
            # exec(test)는 끝났으므로 '테스트 정의 코드 실행' 단계 자체는 완료됨.
            # 다만 정의된 결과물이 기대한 unittest class 구조가 아님.
            setup_exec_passed=True,
            test_exec_passed=False,
            passed=False,
            failed_stage="define_test",
            error_type="InvalidTestCases",
            error="'TestCases' exists after exec(test) but is not a class",
            output=stdout_buffer.getvalue(),
        )

    # TestCases가 class이긴 한데 unittest.TestCase를 상속하지 않은 경우
    # unittest runner가 이걸 테스트로 인식하지 못함
    if not issubclass(test_cls, unittest.TestCase):
        return BigCodeExecutionTrace(
            code_exec_passed=True,
            # setup_exec_passed=True:
            # test 정의 코드는 실행됐지만, unittest가 해석 가능한 테스트 클래스로
            # 정의되지 않았으므로 define_test 단계 실패로 본다.
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
        # (1) TestCases 클래스로부터 unittest 테스트 묶음(suite) 생성
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_cls)

        # (2) 테스트가 0개면 비정상으로 간주
        # - TestCases 클래스는 존재하지만 test_* 메서드가 하나도 없는 경우
        # - 실제 테스트가 전혀 수행되지 않으므로 통과로 보면 안 됨
        if suite.countTestCases() == 0:
            return BigCodeExecutionTrace(
                code_exec_passed=True,
                # BigCode에서는 별도 setup 단계가 없으므로,
                # 여기의 True는 test definition 단계가 끝났음을 뜻하는 placeholder다.
                setup_exec_passed=True,
                test_exec_passed=False,
                passed=False,
                failed_stage="define_test",
                error_type="NoTestCases",
                error="No test methods found in TestCases",
                output=stdout_buffer.getvalue(),
            )

        # (3) unittest 실행 결과 로그를 저장할 버퍼
        run_stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=run_stream, verbosity=0)

        # (4) 실제 테스트 실행
        with contextlib.redirect_stdout(stdout_buffer):
            result = runner.run(suite)

        # (5) 테스트 실패/에러가 하나라도 있으면 run_test 실패
        if not result.wasSuccessful():
            details = run_stream.getvalue().strip()

            # multiple failure / error 수집
            error_traces = [tb for _, tb in result.errors]
            failure_traces = [tb for _, tb in result.failures]

            # error_type은 대표 원인 1개를 뽑아 저장
            # - errors가 있으면 runtime error를 우선 대표값으로 사용
            # - 없고 failures만 있으면 첫 번째 assertion failure를 사용
            if error_traces:
                error_type = _extract_error_type(error_traces[0])
            elif failure_traces:
                error_type = _extract_error_type(failure_traces[0])
            else:
                error_type = "TestFailure"

            # 전체 실패 내용을 하나의 문자열로 합쳐 저장
            # 나중에 디버깅할 때 첫 실패만 보는 것보다 훨씬 유용하다.
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
                # setup_exec_passed=True:
                # test 정의 단계는 성공했고, 실제 실패는 테스트 실행(run_test)에서 발생
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
            # test 정의는 끝났고, runner 수행 중 예외가 난 것으로 해석
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
        # BigCode에서 setup_exec_passed는
        # 'test definition 단계 성공'을 의미하는 통합 스키마용 필드
        setup_exec_passed=True,
        test_exec_passed=True,
        passed=True,
        output=stdout_buffer.getvalue(),
    )