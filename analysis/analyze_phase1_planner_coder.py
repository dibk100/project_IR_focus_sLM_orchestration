# python analysis/analyze_phase1_planner_coder.py results/phase1_easy/planner_coder/details.jsonl
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
        default="results/phase1_easy/planner_coder/details.jsonl",
        help="Path to planner_coder details.jsonl",
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
    print(f"Pass@1: {passed / total:.4f}" if total > 0 else "Pass@1: N/A")

    # error distribution
    error_counter = Counter(
        x["error_type"] for x in data if x["error_type"] is not None
    )

    print("\n📉 Error Type Distribution")
    if error_counter:
        for k, v in error_counter.most_common():
            print(f"{k}: {v}")
    else:
        print("(none)")

    # total latency
    total_latencies = [x["latency_sec"] for x in data if x.get("latency_sec") is not None]
    planner_latencies = [x["planner_latency_sec"] for x in data if x.get("planner_latency_sec") is not None]
    coder_latencies = [x["coder_latency_sec"] for x in data if x.get("coder_latency_sec") is not None]

    if total_latencies:
        print("\n⏱ Total Latency")
        print(f"Avg: {statistics.mean(total_latencies):.3f}s")
        print(f"Max: {max(total_latencies):.3f}s")

    if planner_latencies:
        print("\n🧠 Planner Latency")
        print(f"Avg: {statistics.mean(planner_latencies):.3f}s")
        print(f"Max: {max(planner_latencies):.3f}s")

    if coder_latencies:
        print("\n💻 Coder Latency")
        print(f"Avg: {statistics.mean(coder_latencies):.3f}s")
        print(f"Max: {max(coder_latencies):.3f}s")

    # planner output length stats
    planner_lengths = [len(x.get("planner_output", "")) for x in data]
    if planner_lengths:
        print("\n📝 Planner Output Length")
        print(f"Avg chars: {statistics.mean(planner_lengths):.1f}")
        print(f"Max chars: {max(planner_lengths)}")

    # raw coder output length stats
    coder_lengths = [len(x.get("raw_output", "")) for x in data]
    if coder_lengths:
        print("\n📦 Raw Coder Output Length")
        print(f"Avg chars: {statistics.mean(coder_lengths):.1f}")
        print(f"Max chars: {max(coder_lengths)}")

    # sample failures
    print("\n❌ Sample Failures (up to 3)")
    failures = [x for x in data if x["status"] == "fail"][:3]
    if not failures:
        print("No failures found.")
    else:
        for f in failures:
            print("-" * 40)
            print(f"Task: {f['task_id']}")
            print(f"Error Type: {f['error_type']}")
            print(f"Error Msg: {f['error_message']}")

    # sample planner / coder pairs
    print("\n🧪 Sample Planner/Coder Outputs (up to 2)")
    samples = data[:2]
    for s in samples:
        print("=" * 80)
        print(f"Task: {s['task_id']}")
        print("\n[PLANNER OUTPUT]")
        print(s.get("planner_output", "")[:600])
        print("\n[RAW CODER OUTPUT]")
        print(s.get("raw_output", "")[:600])
        print("\n[GENERATED CODE]")
        print(s.get("generated_code", "")[:600])
        print(f"\n[STATUS] {s['status']} / {s['error_type']}")

    # simple heuristics for planner quality
    print("\n🔍 Planner Output Heuristics")
    added_assumption_count = 0
    code_leak_count = 0

    for row in data:
        planner_output = row.get("planner_output", "").lower()
        if "def " in planner_output or "return " in planner_output:
            code_leak_count += 1

        if (
            "must be non-negative" in planner_output
            or "raise valueerror" in planner_output
            or "epsilon" in planner_output
        ):
            added_assumption_count += 1

    print(f"Planner outputs containing code-like text: {code_leak_count}")
    print(f"Planner outputs with possible extra assumptions: {added_assumption_count}")


if __name__ == "__main__":
    main()