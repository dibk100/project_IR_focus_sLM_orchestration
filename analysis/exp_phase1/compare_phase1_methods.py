# python analysis/exp_phase1/compare_phase1_methods.py \
#   results/phase1_easy/single/details.jsonl \
#   results/phase1_easy/refinement/details.jsonl \
#   results/phase1_easy/repair/details.jsonl \
#   results/phase1_easy/planner_coder/details.jsonl \
#   results/phase1_easy/verification/details.jsonl

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


def build_iterative_final_rows(rows, method_name):
    grouped = group_by_task(rows)
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
            "method": method_name,
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


def build_verification_generation_rows(verification_rows):
    """
    verification 실험의 'generation 성능'만 다른 방법과 비교 가능하도록 정리
    """
    final_rows = []
    for r in verification_rows:
        final_rows.append({
            "task_id": r["task_id"],
            "method": "verification_gen",
            "initial_status": r["status"],
            "initial_error_type": r["error_type"],
            "final_status": r["status"],
            "final_error_type": r["error_type"],
            "passed": r["status"] == "pass",
            "timeout": r["status"] == "timeout",
            "solved_on_attempt": 0 if r["status"] == "pass" else None,
            "num_attempts": 1,
            "latency_sec_sum": r.get("latency_sec"),
            "verifier_correct": r.get("verifier_correct"),
            "verifier_verdict": r.get("verifier_verdict"),
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


def compare_gain(single_rows, target_rows, target_name):
    single_map = {r["task_id"]: r for r in single_rows}
    target_map = {r["task_id"]: r for r in target_rows}

    gain = []
    for task_id, s in single_map.items():
        t = target_map.get(task_id)
        if t is None:
            continue
        if (not s["passed"]) and t["passed"]:
            gain.append(task_id)

    print("\n" + "=" * 60)
    print(f"Gain vs Single: {target_name}")
    print(f"Gain cases: {len(gain)}")
    print("Sample gain tasks:", gain[:10])

    return set(gain)


def print_overlap(single_rows, refinement_rows, repair_rows, planner_rows):
    single_map = {r["task_id"]: r for r in single_rows}
    refinement_map = {r["task_id"]: r for r in refinement_rows}
    repair_map = {r["task_id"]: r for r in repair_rows}
    planner_map = {r["task_id"]: r for r in planner_rows}

    refinement_gain = set()
    repair_gain = set()
    planner_gain = set()

    for task_id, s in single_map.items():
        if (not s["passed"]) and refinement_map.get(task_id, {}).get("passed", False):
            refinement_gain.add(task_id)
        if (not s["passed"]) and repair_map.get(task_id, {}).get("passed", False):
            repair_gain.add(task_id)
        if (not s["passed"]) and planner_map.get(task_id, {}).get("passed", False):
            planner_gain.add(task_id)

    print("\n" + "=" * 60)
    print("Gain Overlap (vs Single)")
    print(f"Refinement gain: {len(refinement_gain)}")
    print(f"Repair gain: {len(repair_gain)}")
    print(f"Planner gain: {len(planner_gain)}")

    print(f"\nRefinement-only gain: {len(refinement_gain - repair_gain - planner_gain)}")
    print(f"Repair-only gain: {len(repair_gain - refinement_gain - planner_gain)}")
    print(f"Planner-only gain: {len(planner_gain - refinement_gain - repair_gain)}")

    print(f"\nRefinement ∩ Repair: {len(refinement_gain & repair_gain)}")
    print(f"Refinement ∩ Planner: {len(refinement_gain & planner_gain)}")
    print(f"Repair ∩ Planner: {len(repair_gain & planner_gain)}")
    print(f"All three: {len(refinement_gain & repair_gain & planner_gain)}")

    print("\nSample Planner-only gain tasks:", sorted(list(planner_gain - refinement_gain - repair_gain))[:10])
    print("Sample Repair-only gain tasks:", sorted(list(repair_gain - refinement_gain - planner_gain))[:10])
    print("Sample Refinement-only gain tasks:", sorted(list(refinement_gain - repair_gain - planner_gain))[:10])


def print_verification_summary(verification_rows):
    total = len(verification_rows)
    verifier_correct = sum(1 for r in verification_rows if r.get("verifier_correct") is True)
    verifier_accuracy = verifier_correct / total if total > 0 else 0.0

    tp = fp = tn = fn = 0
    for r in verification_rows:
        gt_pass = r["passed"]
        pred_pass = r.get("verifier_verdict") == "PASS"

        if gt_pass and pred_pass:
            tp += 1
        elif (not gt_pass) and pred_pass:
            fp += 1
        elif (not gt_pass) and (not pred_pass):
            tn += 1
        elif gt_pass and (not pred_pass):
            fn += 1

    print("\n" + "=" * 60)
    print("Verification Primitive Summary")
    print(f"Verifier Accuracy: {verifier_accuracy:.4f}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")


def print_task_examples(single_rows, refinement_rows, repair_rows, planner_rows, verification_rows, limit=10):
    single_map = {r["task_id"]: r for r in single_rows}
    refinement_map = {r["task_id"]: r for r in refinement_rows}
    repair_map = {r["task_id"]: r for r in repair_rows}
    planner_map = {r["task_id"]: r for r in planner_rows}
    verification_map = {r["task_id"]: r for r in verification_rows}

    print("\n" + "=" * 60)
    print(f"Task-level Comparison Samples (up to {limit})")

    shown = 0
    for task_id in sorted(single_map.keys()):
        s = single_map[task_id]
        rf = refinement_map.get(task_id)
        rp = repair_map.get(task_id)
        p = planner_map.get(task_id)
        v = verification_map.get(task_id)

        print("-" * 60)
        print(f"Task: {task_id}")
        print(f"  single           : {s['final_status']} ({s['final_error_type']})")
        print(f"  refinement       : {rf['final_status']} ({rf['final_error_type']})")
        print(f"  repair           : {rp['final_status']} ({rp['final_error_type']})")
        print(f"  planner_coder    : {p['final_status']} ({p['final_error_type']})")
        print(
            f"  verification_gen : {v['final_status']} ({v['final_error_type']}) | "
            f"verdict={v.get('verifier_verdict')}"
        )

        shown += 1
        if shown >= limit:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("single_path", type=str)
    parser.add_argument("refinement_path", type=str)
    parser.add_argument("repair_path", type=str)
    parser.add_argument("planner_path", type=str)
    parser.add_argument("verification_path", type=str)
    args = parser.parse_args()

    single_raw = load_jsonl(args.single_path)
    refinement_raw = load_jsonl(args.refinement_path)
    repair_raw = load_jsonl(args.repair_path)
    planner_raw = load_jsonl(args.planner_path)
    verification_raw = load_jsonl(args.verification_path)

    single_final = build_single_final_rows(single_raw)
    refinement_final = build_iterative_final_rows(refinement_raw, "refinement")
    repair_final = build_iterative_final_rows(repair_raw, "repair")
    planner_final = build_planner_final_rows(planner_raw)
    verification_final = build_verification_generation_rows(verification_raw)

    print("📂 Input Files")
    print(f"  single          : {args.single_path}")
    print(f"  refinement      : {args.refinement_path}")
    print(f"  repair          : {args.repair_path}")
    print(f"  planner_coder   : {args.planner_path}")
    print(f"  verification    : {args.verification_path}")

    single_summary = summarize_method(single_final)
    refinement_summary = summarize_method(refinement_final)
    repair_summary = summarize_method(repair_final)
    planner_summary = summarize_method(planner_final)
    verification_summary = summarize_method(verification_final)

    print_method_summary("single", single_summary)
    print_method_summary("refinement", refinement_summary)
    print_method_summary("repair", repair_summary)
    print_method_summary("planner_coder", planner_summary)
    print_method_summary("verification_gen", verification_summary)

    compare_gain(single_final, refinement_final, "refinement")
    compare_gain(single_final, repair_final, "repair")
    compare_gain(single_final, planner_final, "planner_coder")

    print_overlap(single_final, refinement_final, repair_final, planner_final)
    print_verification_summary(verification_raw)
    print_task_examples(
        single_final,
        refinement_final,
        repair_final,
        planner_final,
        verification_final,
        limit=10,
    )


if __name__ == "__main__":
    main()