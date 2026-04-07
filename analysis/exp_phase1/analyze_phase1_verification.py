# python analysis/exp_phase1/analyze_phase1_verification.py results/phase1_easy/verification/details.jsonl
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


def compute_confusion_counts(data):
    tp = fp = tn = fn = 0

    for row in data:
        gt_pass = row["passed"]
        pred_pass = row["verifier_verdict"] == "PASS"

        if gt_pass and pred_pass:
            tp += 1
        elif not gt_pass and pred_pass:
            fp += 1
        elif not gt_pass and not pred_pass:
            tn += 1
        elif gt_pass and not pred_pass:
            fn += 1

    return tp, fp, tn, fn


def safe_div(num, den):
    return num / den if den != 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="results/phase1_easy/verification/details.jsonl",
        help="Path to verification details.jsonl",
    )
    args = parser.parse_args()

    data = load_results(args.path)

    total = len(data)

    # generation summary
    passed = sum(1 for x in data if x["status"] == "pass")
    timeout = sum(1 for x in data if x["status"] == "timeout")
    failed = total - passed - timeout

    # verifier summary
    verifier_correct = sum(1 for x in data if x["verifier_correct"])
    verifier_accuracy = safe_div(verifier_correct, total)

    tp, fp, tn, fn = compute_confusion_counts(data)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    print("=" * 60)
    print(f"📂 File: {args.path}")
    print("📊 Generation Summary")
    print(f"Total: {total}")
    print(f"Pass: {passed}")
    print(f"Fail: {failed}")
    print(f"Timeout: {timeout}")
    print(f"Pass@1: {safe_div(passed, total):.4f}")

    # generation error distribution
    error_counter = Counter(
        x["error_type"] for x in data if x["error_type"] is not None
    )

    print("\n📉 Generation Error Type Distribution")
    if error_counter:
        for k, v in error_counter.most_common():
            print(f"{k}: {v}")
    else:
        print("(none)")

    # verifier metrics
    print("\n🔍 Verification Summary")
    print(f"Verifier Accuracy: {verifier_accuracy:.4f}")
    print(f"TP (correct PASS): {tp}")
    print(f"FP (wrong PASS):   {fp}")
    print(f"TN (correct FAIL): {tn}")
    print(f"FN (wrong FAIL):   {fn}")
    print(f"Precision (PASS):  {precision:.4f}")
    print(f"Recall (PASS):     {recall:.4f}")
    print(f"Specificity (FAIL): {specificity:.4f}")
    print(f"F1 (PASS):         {f1:.4f}")

    # verifier verdict distribution
    verdict_counter = Counter(x["verifier_verdict"] for x in data)

    print("\n🗳 Verifier Verdict Distribution")
    for k, v in verdict_counter.items():
        print(f"{k}: {v}")

    # latency
    total_latencies = [x["latency_sec"] for x in data if x.get("latency_sec") is not None]
    gen_latencies = [x["generator_latency_sec"] for x in data if x.get("generator_latency_sec") is not None]
    ver_latencies = [x["verifier_latency_sec"] for x in data if x.get("verifier_latency_sec") is not None]

    if total_latencies:
        print("\n⏱ Total Latency")
        print(f"Avg: {statistics.mean(total_latencies):.3f}s")
        print(f"Max: {max(total_latencies):.3f}s")

    if gen_latencies:
        print("\n⚙ Generator Latency")
        print(f"Avg: {statistics.mean(gen_latencies):.3f}s")
        print(f"Max: {max(gen_latencies):.3f}s")

    if ver_latencies:
        print("\n🧪 Verifier Latency")
        print(f"Avg: {statistics.mean(ver_latencies):.3f}s")
        print(f"Max: {max(ver_latencies):.3f}s")

    # sample wrong verifier cases
    print("\n❌ Sample Wrong Verification Cases (up to 5)")
    wrong_cases = [x for x in data if not x["verifier_correct"]][:5]
    if not wrong_cases:
        print("No wrong verification cases found.")
    else:
        for row in wrong_cases:
            gt = "PASS" if row["passed"] else "FAIL"
            print("-" * 40)
            print(f"Task: {row['task_id']}")
            print(f"Ground Truth: {gt}")
            print(f"Verifier Verdict: {row['verifier_verdict']}")
            print(f"Error Type: {row['error_type']}")
            print(f"Verifier Output: {row['verifier_output'][:500]}")

    # sample correct fail-detection cases
    print("\n✅ Sample Correct FAIL Detection Cases (up to 3)")
    correct_fail_cases = [
        x for x in data
        if (not x["passed"]) and x["verifier_verdict"] == "FAIL"
    ][:3]

    if not correct_fail_cases:
        print("No correct FAIL detections found.")
    else:
        for row in correct_fail_cases:
            print("-" * 40)
            print(f"Task: {row['task_id']}")
            print(f"Error Type: {row['error_type']}")
            print(f"Verifier Output: {row['verifier_output'][:500]}")

    # sample missed fail cases (false positives)
    print("\n⚠ Sample Missed FAIL Cases / False Positives (up to 3)")
    false_positive_cases = [
        x for x in data
        if (not x["passed"]) and x["verifier_verdict"] == "PASS"
    ][:3]

    if not false_positive_cases:
        print("No false positives found.")
    else:
        for row in false_positive_cases:
            print("-" * 40)
            print(f"Task: {row['task_id']}")
            print(f"Error Type: {row['error_type']}")
            print(f"Verifier Output: {row['verifier_output'][:500]}")


if __name__ == "__main__":
    main()