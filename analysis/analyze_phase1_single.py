# python analysis/analyze_phase1_single.py results/phase1_easy/single/details.jsonl
import json
import argparse
from collections import Counter
import statistics


def load_results(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=" ",
        help="Path to details.jsonl",
    )
    args = parser.parse_args()

    data = load_results(args.path)

    total = len(data)
    passed = sum(1 for x in data if x["status"] == "pass")
    timeout = sum(1 for x in data if x["status"] == "timeout")
    failed = total - passed - timeout

    print("=" * 60)
    print(f"📂 File: {args.path}")
    print("📊 Overall")
    print(f"Total: {total}")
    print(f"Pass: {passed}")
    print(f"Fail: {failed}")
    print(f"Timeout: {timeout}")
    print(f"Pass@1: {passed / total:.4f}")

    # error distribution
    error_counter = Counter(
        x["error_type"] for x in data if x["error_type"] is not None
    )

    print("\n📉 Error Type Distribution")
    for k, v in error_counter.most_common():
        print(f"{k}: {v}")

    # latency
    latencies = [x["latency_sec"] for x in data if x["latency_sec"] is not None]
    if latencies:
        print("\n⏱ Latency")
        print(f"Avg: {statistics.mean(latencies):.3f}s")
        print(f"Max: {max(latencies):.3f}s")

    # sample failures
    print("\n❌ Sample Failures (up to 3)")
    failures = [x for x in data if x["status"] == "fail"][:3]
    for f in failures:
        print("-" * 40)
        print(f"Task: {f['task_id']}")
        print(f"Error Type: {f['error_type']}")
        print(f"Error Msg: {f['error_message']}")


if __name__ == "__main__":
    main()