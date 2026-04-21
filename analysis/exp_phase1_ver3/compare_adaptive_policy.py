"""
4단계 비교 분석 스크립트: adaptive_policy vs single / retry / repair / planner+repair
===================================================================================

사용법:
    # 프로젝트 루트에서
    PYTHONPATH=. python analysis/exp_phase1_ver3/compare_adaptive_policy.py \
        --dataset humaneval

출력:
    - 콘솔 표 (pass@1, exec_rate, avg_calls, avg_tokens 비교)
    - results/phase1_ver3/<dataset>/comparison_adaptive_policy.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

# ─── 설정 ────────────────────────────────────────
RESULT_BASE = "results/phase1_ver3"

# dataset별 방법 → 결과 디렉토리 매핑
METHOD_DIR_MAP: dict[str, str] = {
    "single":             "single",
    "retry":              "retry",
    "repair":             "repair",
    "planner_repair":     "planner_coder_repair",
    "adaptive_policy":    "adaptive_policy",
}

SUMMARY_FILE  = "summary.json"
ANALYSIS_FILE = "analysis.json"


# ─── 로드 헬퍼 ───────────────────────────────────

def load_summary(dataset: str, method_dir: str) -> Optional[dict]:
    path = os.path.join(RESULT_BASE, dataset, method_dir, SUMMARY_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_step_logs(dataset: str, method_dir: str) -> list[dict]:
    path = os.path.join(RESULT_BASE, dataset, method_dir, "step_logs.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trajectory_logs(dataset: str, method_dir: str) -> list[dict]:
    path = os.path.join(RESULT_BASE, dataset, method_dir, "trajectory_logs.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─── 비교 지표 계산 ──────────────────────────────

def compute_metrics(dataset: str, method_key: str, method_dir: str) -> dict:
    summary = load_summary(dataset, method_dir)
    if summary is None:
        return {"method": method_key, "available": False}

    trajs = load_trajectory_logs(dataset, method_dir)

    avg_calls  = summary.get("avg_calls", 0.0)
    avg_tokens = summary.get("avg_tokens", 0.0)

    # adaptive_policy 전용: action 분포
    action_dist: dict[str, int] = {}
    if method_key == "adaptive_policy":
        for t in trajs:
            for a in t.get("action_history", []):
                action_dist[a] = action_dist.get(a, 0) + 1

    return {
        "method":                 method_key,
        "available":              True,
        "pass_at_1":              summary.get("pass_at_1", 0.0),
        "execution_success_rate": summary.get("execution_success_rate", 0.0),
        "conditional_pass":       summary.get("conditional_pass", 0.0),
        "avg_calls":              avg_calls,
        "avg_tokens":             avg_tokens,
        "num_pass":               summary.get("num_pass", 0),
        "total_problems":         summary.get("total_problems", 0),
        "action_dist":            action_dist,   # adaptive_policy 전용
    }


# ─── 출력 포맷 ───────────────────────────────────

HEADER = (
    f"{'Method':<22} "
    f"{'pass@1':>8} "
    f"{'exec_rate':>10} "
    f"{'cond_pass':>10} "
    f"{'avg_calls':>10} "
    f"{'avg_tokens':>11}"
)
SEP = "-" * len(HEADER)


def print_row(m: dict) -> None:
    if not m["available"]:
        print(f"  {m['method']:<20}  결과 없음 (실험 미실행)")
        return
    print(
        f"  {m['method']:<20} "
        f"  {m['pass_at_1']:>7.4f} "
        f"  {m['execution_success_rate']:>9.4f} "
        f"  {m['conditional_pass']:>9.4f} "
        f"  {m['avg_calls']:>9.2f} "
        f"  {m['avg_tokens']:>10.1f}"
    )


# ─── 메인 ────────────────────────────────────────

def main(dataset: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  📊 4단계 비교: adaptive_policy vs 기존 방법  [dataset={dataset}]")
    print(f"{'=' * 70}")
    print(f"  {HEADER}")
    print(f"  {SEP}")

    all_metrics = []
    for method_key, method_dir in METHOD_DIR_MAP.items():
        m = compute_metrics(dataset, method_key, method_dir)
        all_metrics.append(m)
        print_row(m)

    print(f"  {SEP}")

    # adaptive_policy 전용 action 분포 출력
    ap = next((m for m in all_metrics if m["method"] == "adaptive_policy"), None)
    if ap and ap.get("available") and ap.get("action_dist"):
        print(f"\n  📌 Adaptive Policy Action 분포 [{dataset}]:")
        for action, cnt in sorted(ap["action_dist"].items(), key=lambda x: -x[1]):
            print(f"    {action:<12}: {cnt}")

    # 결과 저장
    out_dir  = os.path.join(RESULT_BASE, dataset)
    out_path = os.path.join(out_dir, "comparison_adaptive_policy.json")
    os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset": dataset, "methods": all_metrics},
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n  💾 비교 결과 저장: {out_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp", "bigcode"],
        help="비교할 데이터셋 이름",
    )
    args = parser.parse_args()
    main(args.dataset)
