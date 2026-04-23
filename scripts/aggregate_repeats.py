"""
python scripts/aggregate_repeats.py \
  --run_dirs \
    results/phase2_qwen/humaneval/rpe_policy_seed42_repeat1 \
    results/phase2_qwen/humaneval/rpe_policy_seed42_repeat2 \
    results/phase2_qwen/humaneval/rpe_policy_seed42_repeat3 \
    results/phase2_qwen/humaneval/rpe_policy_seed42_repeat4 \
    results/phase2_qwen/humaneval/rpe_policy_seed42_repeat5 \
  --output results/phase2_qwen/humaneval/rpe_policy_seed42_aggregate.json
"""

import argparse
import json
import math
from pathlib import Path

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_summary(run_dir):
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_metric(summaries, key):
    return [s[key] for s in summaries]


def collect_nested_metric(summaries, parent, key):
    return [s[parent][key] for s in summaries]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True, help="List of run output directories")
    parser.add_argument("--output", required=True, help="Path to save aggregated JSON")
    args = parser.parse_args()

    summaries = [load_summary(run_dir) for run_dir in args.run_dirs]

    pass_at_1 = collect_metric(summaries, "pass_at_1")
    execution_success_rate = collect_metric(summaries, "execution_success_rate")
    conditional_pass = collect_metric(summaries, "conditional_pass")
    num_pass = collect_metric(summaries, "num_pass")

    code_failed = collect_nested_metric(summaries, "extra_summary", "code_failed")
    define_test_failed = collect_nested_metric(summaries, "extra_summary", "define_test_failed")
    run_test_failed = collect_nested_metric(summaries, "extra_summary", "run_test_failed")

    result = {
        "num_runs": len(summaries),
        "pass_at_1": {
            "mean": mean(pass_at_1),
            "std": std(pass_at_1),
            "values": pass_at_1,
        },
        "execution_success_rate": {
            "mean": mean(execution_success_rate),
            "std": std(execution_success_rate),
            "values": execution_success_rate,
        },
        "conditional_pass": {
            "mean": mean(conditional_pass),
            "std": std(conditional_pass),
            "values": conditional_pass,
        },
        "num_pass": {
            "mean": mean(num_pass),
            "std": std(num_pass),
            "values": num_pass,
        },
        "code_failed": {
            "mean": mean(code_failed),
            "std": std(code_failed),
            "values": code_failed,
        },
        "define_test_failed": {
            "mean": mean(define_test_failed),
            "std": std(define_test_failed),
            "values": define_test_failed,
        },
        "run_test_failed": {
            "mean": mean(run_test_failed),
            "std": std(run_test_failed),
            "values": run_test_failed,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()