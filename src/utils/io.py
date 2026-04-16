import os
import json
from datetime import datetime
from typing import Any, Dict, List

def make_run_id(config: dict) -> str:
    """
    run_id 자동 생성/보정
    - config에 값이 있으면 기본값으로 사용
    - 뒤에 timestamp를 붙여 고유하게 만듦
    """
    base_run_id = config.get("run", {}).get("run_id", "phase1_single")
    suffix = datetime.now().strftime("%m%d%H%M%S")
    return f"{base_run_id}_{suffix}"

def save_result(result: Dict[str, Any], output_path: str) -> None:
    """단일 결과 JSON 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 결과 저장: {output_path}")


def save_results_jsonl(results: List[Dict[str, Any]], output_path: str) -> None:
    """리스트를 JSONL로 저장 (overwrite)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    print(f"💾 결과 저장: {output_path} ({len(results)}건)")


def append_jsonl(record: Dict[str, Any], output_path: str) -> None:
    """JSONL 한 줄 append 저장 (streaming log용)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def save_config(config: Dict[str, Any], output_dir: str) -> None:
    """config snapshot 저장"""
    path = os.path.join(output_dir, "config.json")
    save_result(config, path)