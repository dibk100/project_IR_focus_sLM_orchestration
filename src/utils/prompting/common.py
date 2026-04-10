# src/utils/prompting/base.py
"""
공통 프롬프트/생성 후처리 유틸리티
"""
import re

def strip_code_fence(text: str) -> str:
    """
    # markdown code fence 최소 제거
    """
    text = text.replace("```python", "")
    text = text.replace("```", "")
    return text.strip("\n")


def truncate_at_new_toplevel_block(text: str) -> str:
    """
    생성된 코드에서 첫 번째 top-level 블록 이후에
    새로운 def/class/if __name__ 블록이 나오면 거기서 자르기

    HumanEval처럼 '함수 본문만 이어서 생성'하는 경우에 유용
    """
    lines = text.split("\n")
    kept_lines = []

    for line in lines:
        # 새로운 top-level 정의가 나오면 중단
        if kept_lines and (
            line.startswith("def ")
            or line.startswith("class ")
            or line.startswith("if __name__")
        ):
            break

        kept_lines.append(line)

    return "\n".join(kept_lines).rstrip()