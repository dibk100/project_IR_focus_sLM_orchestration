# src/utils/io.py

"""
결과 저장 유틸리티
실험 결과를 JSON/JSONL로 저장한다.
"""
import os
import json
from datetime import datetime
from typing import Any, Dict, List


def save_result(result: Dict[str, Any], output_path: str) -> None:
    """단일 결과를 JSON으로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 결과 저장: {output_path}")


def save_results_jsonl(results: List[Dict[str, Any]], output_path: str) -> None:
    """결과 리스트를 JSONL로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    print(f"💾 결과 저장: {output_path} ({len(results)}건)")


def make_output_dir(base_dir: str, experiment_name: str) -> str:
    """타임스탬프 기반 결과 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
