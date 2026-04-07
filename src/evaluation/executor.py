"""
코드 실행기
생성된 코드 + 테스트 케이스를 안전하게 실행하여 pass/fail을 판정한다.
"""
import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """코드 실행 결과"""
    passed: bool
    output: str
    error: Optional[str] = None
    timeout: bool = False


def execute_code(code: str, test: str, entry_point: str, timeout: int = 10) -> ExecutionResult:
    """생성된 코드 + 테스트를 실행하여 결과 반환
    
    Args:
        code: 완성된 함수 코드
        test: HumanEval 테스트 코드 (check 함수 포함)
        entry_point: 함수 이름
        timeout: 실행 제한 시간 (초)
    
    Returns:
        ExecutionResult
    """
    # 실행할 전체 코드 구성: 함수 정의 + 테스트
    full_code = code + "\n" + test + f"\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
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
            return ExecutionResult(passed=True, output=result.stdout)
        else:
            return ExecutionResult(
                passed=False, output=result.stdout, error=result.stderr
            )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False, output="", error="Timeout", timeout=True
        )
    except Exception as e:
        return ExecutionResult(passed=False, output="", error=str(e))
    finally:
        os.unlink(tmp_path)
