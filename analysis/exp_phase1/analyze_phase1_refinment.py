# python analysis/exp_phase1/analyze_phase1_refinment.py results/phase1_easy/refinement/details.jsonl
import json
import argparse
from collections import Counter, defaultdict
import statistics


def load_results(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def group_by_task(data):
    grouped = defaultdict(list)
    for row in data:
        grouped[row["task_id"]].append(row)

    for task_id in grouped:
        grouped[task_id] = sorted(grouped[task_id], key=lambda x: x["attempt_idx"])
    return grouped


def summarize_task_level(grouped):
    total = len(grouped)

    final_pass = 0
    final_timeout = 0
    final_fail = 0

    solved_on_attempt = Counter()
    refinement_gain = 0

    for task_id, attempts in grouped.items():
        first = attempts[0]
        last = attempts[-1]

        # final result
        if last["status"] == "pass":
            final_pass += 1
        elif last["status"] == "timeout":
            final_timeout += 1
        else:
            final_fail += 1

        # first successful attempt
        first_success_attempt = None
        for row in attempts:
            if row["status"] == "pass":
                first_success_attempt = row["attempt_idx"]
                break

        if first_success_attempt is not None:
            solved_on_attempt[first_success_attempt] += 1

        # refinement gain: initial fail -> final pass
        if first["status"] != "pass" and last["status"] == "pass":
            refinement_gain += 1

    return {
        "total": total,
        "final_pass": final_pass,
        "final_fail": final_fail,
        "final_timeout": final_timeout,
        "final_success_rate": final_pass / total if total > 0 else 0.0,
        "solved_on_attempt": solved_on_attempt,
        "refinement_gain": refinement_gain,
    }


def summarize_attempt_level(data):
    by_attempt = defaultdict(list)
    for row in data:
        by_attempt[row["attempt_idx"]].append(row)

    summary = {}

    for attempt_idx, rows in sorted(by_attempt.items()):
        total = len(rows)
        passed = sum(1 for r in rows if r["status"] == "pass")
        timeout = sum(1 for r in rows if r["status"] == "timeout")
        failed = total - passed - timeout

        error_counter = Counter(
            r["error_type"] for r in rows
            if r["status"] != "pass" and r["error_type"] is not None
        )

        latencies = [r["latency_sec"] for r in rows if r["latency_sec"] is not None]

        summary[attempt_idx] = {
            "total": total,
            "pass": passed,
            "fail": failed,
            "timeout": timeout,
            "pass_rate": passed / total if total > 0 else 0.0,
            "error_counter": error_counter,
            "avg_latency": statistics.mean(latencies) if latencies else None,
            "max_latency": max(latencies) if latencies else None,
        }

    return summary


def print_task_level_summary(task_summary):
    print("=" * 60)
    print("📊 Task-level Final Summary")
    print(f"Total tasks: {task_summary['total']}")
    print(f"Final Pass: {task_summary['final_pass']}")
    print(f"Final Fail: {task_summary['final_fail']}")
    print(f"Final Timeout: {task_summary['final_timeout']}")
    print(f"Final Success Rate: {task_summary['final_success_rate']:.4f}")
    print(f"Refinement Gain (initial fail -> final pass): {task_summary['refinement_gain']}")

    print("\n🎯 Solved on Attempt")
    for attempt_idx, count in sorted(task_summary["solved_on_attempt"].items()):
        print(f"Attempt {attempt_idx}: {count}")


def print_attempt_level_summary(attempt_summary):
    print("\n" + "=" * 60)
    print("📈 Attempt-level Summary")

    for attempt_idx, s in attempt_summary.items():
        print("\n" + "-" * 40)
        print(f"Attempt {attempt_idx}")
        print(f"Total: {s['total']}")
        print(f"Pass: {s['pass']}")
        print(f"Fail: {s['fail']}")
        print(f"Timeout: {s['timeout']}")
        print(f"Pass Rate: {s['pass_rate']:.4f}")

        if s["avg_latency"] is not None:
            print(f"Avg Latency: {s['avg_latency']:.3f}s")
            print(f"Max Latency: {s['max_latency']:.3f}s")

        print("Error Type Distribution:")
        if s["error_counter"]:
            for err, cnt in s["error_counter"].most_common():
                print(f"  {err}: {cnt}")
        else:
            print("  (none)")


def print_sample_refined_cases(grouped, limit=3):
    print("\n" + "=" * 60)
    print(f"🛠 Sample Refined Cases (up to {limit})")

    shown = 0
    for task_id, attempts in grouped.items():
        first = attempts[0]
        last = attempts[-1]

        if first["status"] != "pass" and last["status"] == "pass":
            print("-" * 40)
            print(f"Task: {task_id}")
            print(f"Initial Status: {first['status']} ({first['error_type']})")
            print(f"Final Status: {last['status']}")
            print(f"Solved at attempt: {last['attempt_idx']}")
            shown += 1

            if shown >= limit:
                break

    if shown == 0:
        print("No refined cases found.")


def print_sample_unrefined_cases(grouped, limit=3):
    print("\n" + "=" * 60)
    print(f"❌ Sample Unrefined Cases (up to {limit})")

    shown = 0
    for task_id, attempts in grouped.items():
        first = attempts[0]
        last = attempts[-1]

        if last["status"] != "pass":
            print("-" * 40)
            print(f"Task: {task_id}")
            print(f"Initial Status: {first['status']} ({first['error_type']})")
            print(f"Final Status: {last['status']} ({last['error_type']})")
            print(f"Attempts used: {len(attempts)}")
            print(f"Final Error Msg: {last['error_message']}")
            shown += 1

            if shown >= limit:
                break

    if shown == 0:
        print("All tasks were solved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="results/phase1_easy/refinement/details.jsonl",
        help="Path to refinement details.jsonl",
    )
    args = parser.parse_args()

    data = load_results(args.path)
    grouped = group_by_task(data)

    print(f"📂 File: {args.path}")
    print(f"🧾 Total attempt rows: {len(data)}")
    print(f"🧩 Total unique tasks: {len(grouped)}")

    task_summary = summarize_task_level(grouped)
    attempt_summary = summarize_attempt_level(data)

    print_task_level_summary(task_summary)
    print_attempt_level_summary(attempt_summary)
    print_sample_refined_cases(grouped, limit=3)
    print_sample_unrefined_cases(grouped, limit=3)


if __name__ == "__main__":
    main()