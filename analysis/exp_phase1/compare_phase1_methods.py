# python analysis/compare_phase1_methods.py \
#   results/phase1_easy/single/details.jsonl \
#   results/phase1_easy/repair/details.jsonl \
#   results/phase1_easy/planner_coder/details.jsonl

import json
import argparse
from collections import defaultdict
import statistics


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def group_by_task(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["task_id"]].append(r)
    for task_id in grouped:
        grouped[task_id] = sorted(grouped[task_id], key=lambda x: x["attempt_idx"])
    return grouped


def build_single_final_rows(single_rows):
    final_rows = []
    for r in single_rows:
        final_rows.append({
            "task_id": r["task_id"],
            "method": "single",
            "initial_status": r["status"],
            "initial_error_type": r["error_type"],
            "final_status": r["status"],
            "final_error_type": r["error_type"],
            "passed": r["status"] == "pass",
            "timeout": r["status"] == "timeout",
            "solved_on_attempt": 0 if r["status"] == "pass" else None,
            "num_attempts": 1,
            "latency_sec_sum": r.get("latency_sec"),
        })
    return final_rows


def build_repair_final_rows(repair_rows):
    grouped = group_by_task(repair_rows)
    final_rows = []

    for task_id, attempts in grouped.items():
        first = attempts[0]
        last = attempts[-1]

        solved_on_attempt = None
        for row in attempts:
            if row["status"] == "pass":
                solved_on_attempt = row["attempt_idx"]
                break

        final_rows.append({
            "task_id": task_id,
            "method": "repair",
            "initial_status": first["status"],
            "initial_error_type": first["error_type"],
            "final_status": last["status"],
            "final_error_type": last["error_type"],
            "passed": last["status"] == "pass",
            "timeout": last["status"] == "timeout",
            "solved_on_attempt": solved_on_attempt,
            "num_attempts": len(attempts),
            "latency_sec_sum": sum(
                x["latency_sec"] for x in attempts if x.get("latency_sec") is not None
            ),
        })

    return final_rows


def build_planner_final_rows(planner_rows):
    final_rows = []
    for r in planner_rows:
        final_rows.append({
            "task_id": r["task_id"],
            "method": "planner_coder",
            "initial_status": r["status"],
            "initial_error_type": r["error_type"],
            "final_status": r["status"],
            "final_error_type": r["error_type"],
            "passed": r["status"] == "pass",
            "timeout": r["status"] == "timeout",
            "solved_on_attempt": 0 if r["status"] == "pass" else None,
            "num_attempts": 1,
            "latency_sec_sum": r.get("latency_sec"),
        })
    return final_rows


def summarize_method(rows):
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    timeout = sum(1 for r in rows if r["timeout"])
    failed = total - passed - timeout
    success_rate = passed / total if total > 0 else 0.0

    latencies = [r["latency_sec_sum"] for r in rows if r["latency_sec_sum"] is not None]
    avg_latency = statistics.mean(latencies) if latencies else None

    error_dist = defaultdict(int)
    for r in rows:
        if r["final_error_type"] is not None:
            error_dist[r["final_error_type"]] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "timeout": timeout,
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "error_dist": dict(sorted(error_dist.items(), key=lambda x: -x[1])),
    }


def print_method_summary(method_name, summary):
    print("=" * 60)
    print(f"Method: {method_name}")
    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Timeout: {summary['timeout']}")
    print(f"Success Rate: {summary['success_rate']:.4f}")
    if summary["avg_latency"] is not None:
        print(f"Avg Latency: {summary['avg_latency']:.3f}s")
    print("Error Distribution:")
    if summary["error_dist"]:
        for err, cnt in summary["error_dist"].items():
            print(f"  {err}: {cnt}")
    else:
        print("  (none)")


def compare_gain(single_rows, repair_rows, planner_rows):
    single_map = {r["task_id"]: r for r in single_rows}
    repair_map = {r["task_id"]: r for r in repair_rows}
    planner_map = {r["task_id"]: r for r in planner_rows}

    repair_gain = []
    planner_gain = []
    repair_only = []
    planner_only = []
    both_gain = []

    for task_id, s in single_map.items():
        s_pass = s["passed"]
        r_pass = repair_map.get(task_id, {}).get("passed", False)
        p_pass = planner_map.get(task_id, {}).get("passed", False)

        if not s_pass and r_pass:
            repair_gain.append(task_id)
        if not s_pass and p_pass:
            planner_gain.append(task_id)

        if not s_pass and r_pass and p_pass:
            both_gain.append(task_id)
        elif not s_pass and r_pass and not p_pass:
            repair_only.append(task_id)
        elif not s_pass and p_pass and not r_pass:
            planner_only.append(task_id)

    print("\n" + "=" * 60)
    print("Gain vs Single")
    print(f"Repair gain cases: {len(repair_gain)}")
    print(f"Planner-coder gain cases: {len(planner_gain)}")
    print(f"Both gain: {len(both_gain)}")
    print(f"Repair-only gain: {len(repair_only)}")
    print(f"Planner-only gain: {len(planner_only)}")

    print("\nSample Repair-only gain tasks:", repair_only[:10])
    print("Sample Planner-only gain tasks:", planner_only[:10])
    print("Sample Both-gain tasks:", both_gain[:10])


def print_task_examples(single_rows, repair_rows, planner_rows, limit=10):
    single_map = {r["task_id"]: r for r in single_rows}
    repair_map = {r["task_id"]: r for r in repair_rows}
    planner_map = {r["task_id"]: r for r in planner_rows}

    print("\n" + "=" * 60)
    print(f"Task-level Comparison Samples (up to {limit})")

    shown = 0
    for task_id in sorted(single_map.keys()):
        s = single_map[task_id]
        r = repair_map.get(task_id)
        p = planner_map.get(task_id)

        print("-" * 60)
        print(f"Task: {task_id}")
        print(f"  single         : {s['final_status']} ({s['final_error_type']})")
        print(f"  repair         : {r['final_status']} ({r['final_error_type']})")
        print(f"  planner_coder  : {p['final_status']} ({p['final_error_type']})")

        shown += 1
        if shown >= limit:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("single_path", type=str)
    parser.add_argument("repair_path", type=str)
    parser.add_argument("planner_path", type=str)
    args = parser.parse_args()

    single_raw = load_jsonl(args.single_path)
    repair_raw = load_jsonl(args.repair_path)
    planner_raw = load_jsonl(args.planner_path)

    single_final = build_single_final_rows(single_raw)
    repair_final = build_repair_final_rows(repair_raw)
    planner_final = build_planner_final_rows(planner_raw)

    print("📂 Input Files")
    print(f"  single        : {args.single_path}")
    print(f"  repair        : {args.repair_path}")
    print(f"  planner_coder : {args.planner_path}")

    single_summary = summarize_method(single_final)
    repair_summary = summarize_method(repair_final)
    planner_summary = summarize_method(planner_final)

    print_method_summary("single", single_summary)
    print_method_summary("repair", repair_summary)
    print_method_summary("planner_coder", planner_summary)

    compare_gain(single_final, repair_final, planner_final)
    print_task_examples(single_final, repair_final, planner_final, limit=10)


if __name__ == "__main__":
    main()