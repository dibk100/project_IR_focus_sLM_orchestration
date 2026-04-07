"""
HumanEval 데이터셋 다운로드 스크립트
- Source: openai/openai_humaneval (Hugging Face)
- 저장 경로: datasets/humaneval/raw/

pip install datasets
python src/tasks/download_humaneval.py
"""
import os
import json
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "datasets", "humaneval", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def main():
    print("🔄 HumanEval 데이터셋 다운로드 중...")
    dataset = load_dataset("openai/openai_humaneval", split="test")

    # 전체 데이터셋을 jsonl로 저장
    output_path = os.path.join(RAW_DIR, "humaneval.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 완료! {len(dataset)}개 문제 저장됨 → {output_path}")

if __name__ == "__main__":
    main()
