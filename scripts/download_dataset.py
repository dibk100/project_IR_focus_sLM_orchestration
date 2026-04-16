"""
Generic dataset downloader for code benchmarks
- Supports: HumanEval, HumanEval Pro, MBPP, etc.
- Usage examples:

python script/download_dataset.py --name mbpp_pro

pip install datasets
"""

import os
import json
import argparse
from datasets import load_dataset

# -------------------------
# Dataset registry (핵심)
# -------------------------
DATASET_REGISTRY = {
    "humaneval": {
        "hf_name": "openai/openai_humaneval",
        "split": "test",
        "save_name": "humaneval.jsonl",
    },
    "humaneval_pro": {
        "hf_name": "CodeEval-Pro/humaneval-pro",
        "split": "train",
        "save_name": "humaneval_pro.jsonl",
    },
    "mbpp_pro": {
        "hf_name": "CodeEval-Pro/mbpp-pro",
        "split": "train",
        "save_name": "mbpp_pro.jsonl",
    },
    "mbpp": {
        "hf_name": "google-research-datasets/mbpp",             
        "split": "train",                                       
        "save_name": "mbpp.jsonl",
    },
    "codecontests": {
        "hf_name": "deepmind/code_contests",
        "split": "valid",
        "save_name": "codecontests.jsonl",
    },
    "swebench_lite": {
        "hf_name": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "save_name": "swebench_lite.jsonl",
    },
    "bigcode": {
        "hf_name": "bigcode/bigcodebench",
        "split": "v0.1.0_hf",
        "save_name": "bigcode.jsonl",
    },
    "livecodebench": {
        "hf_name": "livecodebench/code_generation",
        "split": "test",
        "save_name": "livecodebench.jsonl",
    },
    "classeval": {
        "hf_name": "FudanSELab/ClassEval",
        "split": "test",
        "save_name": "classeval.jsonl",
    },
}


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def download_dataset(dataset_key: str):
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"❌ Unknown dataset: {dataset_key}")

    config = DATASET_REGISTRY[dataset_key]

    raw_dir = os.path.join(PROJECT_ROOT, "datasets", dataset_key, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"🔄 Downloading {dataset_key} ...")
    print(f"   ↳ HF: {config['hf_name']} ({config['split']})")

    dataset = load_dataset(config["hf_name"], split=config["split"])

    output_path = os.path.join(raw_dir, config["save_name"])

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    print(f"✅ Done! {len(dataset)} samples saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=DATASET_REGISTRY.keys(),
        help="Dataset name (humaneval / humaneval_pro / mbpp)",
    )
    args = parser.parse_args()

    download_dataset(args.name)


if __name__ == "__main__":
    main()