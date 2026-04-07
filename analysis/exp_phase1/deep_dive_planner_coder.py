# python analysis/exp_phase1/deep_dive_planner_coder.py \
#   results/phase1_easy/single/details.jsonl \
#   results/phase1_easy/repair/details.jsonl \
#   results/phase1_easy/planner_coder/details.jsonl

import json
import re
import ast
import argparse
from collections import defaultdict


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


def build_single_map(rows):
    return {r["task_id"]: r for r in rows}


def build_repair_final_map(rows):
    grouped = group_by_task(rows)
    final_map = {}
    for task_id, attempts in grouped.items():
        final_map[task_id] = attempts[-1]
    return final_map


def build_planner_map(rows):
    return {r["task_id"]: r for r in rows}


def extract_function_signature(code: str, entry_point: str):
    """
    반환:
    {
        "found": bool,
        "args": [arg1, arg2, ...],
        "returns": str | None,
    }
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return {"found": False, "args": [], "returns": None}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            args = [a.arg for a in node.args.args]
            returns = None
            if node.returns is not None:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:
                    returns = None
            return {"found": True, "args": args, "returns": returns}

    return {"found": False, "args": [], "returns": None}


def extract_def_names(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
    return names


def extract_assigned_names(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    assigned = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assigned |= extract_target_names(target)
        elif isinstance(node, ast.AnnAssign):
            assigned |= extract_target_names(node.target)
        elif isinstance(node, ast.AugAssign):
            assigned |= extract_target_names(node.target)
        elif isinstance(node, ast.For):
            assigned |= extract_target_names(node.target)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    assigned |= extract_target_names(item.optional_vars)
        elif isinstance(node, ast.comprehension):
            assigned |= extract_target_names(node.target)

    return assigned


def extract_target_names(target):
    names = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names |= extract_target_names(elt)
    return names


def extract_used_names(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return used


def extract_imported_names(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    return imported


def extract_param_names(code: str, entry_point: str):
    sig = extract_function_signature(code, entry_point)
    return set(sig["args"]) if sig["found"] else set()


def infer_undefined_names(code: str, entry_point: str):
    builtin_allow = {
        "len", "range", "enumerate", "abs", "min", "max", "sum", "sorted",
        "list", "set", "dict", "tuple", "str", "int", "float", "bool",
        "zip", "map", "filter", "any", "all", "print"
    }

    used = extract_used_names(code)
    assigned = extract_assigned_names(code)
    imported = extract_imported_names(code)
    params = extract_param_names(code, entry_point)
    defs = extract_def_names(code)

    defined = assigned | imported | params | defs | builtin_allow
    undefined = sorted(x for x in used if x not in defined)
    return undefined


def planner_mentions_suspicious_names(planner_output: str, code: str, entry_point: str):
    """
    planner에서 새 이름/개념을 도입했는지 아주 거칠게 본다.
    """
    planner_text = planner_output or ""
    planner_tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", planner_text))

    code_defined = (
        extract_def_names(code)
        | extract_assigned_names(code)
        | extract_imported_names(code)
        | extract_param_names(code, entry_point)
    )

    generic_words = {
        "the", "a", "an", "and", "or", "if", "else", "for", "while", "return",
        "task", "plan", "edge", "cases", "constraints", "input", "output",
        "string", "list", "number", "numbers", "result", "results",
        "function", "algorithm", "iterate", "sort", "sorted", "true", "false",
        "none", "python", "use", "using", "check", "value", "values",
        "index", "indices", "group", "groups", "count", "counter",
        "threshold", "elements", "element", "char", "characters"
    }

    suspicious = sorted(
        tok for tok in planner_tokens
        if tok not in code_defined
        and tok.lower() not in generic_words
        and tok != entry_point
    )
    return suspicious[:20]


def compare_signature(single_code: str, planner_code: str, entry_point: str):
    s = extract_function_signature(single_code, entry_point)
    p = extract_function_signature(planner_code, entry_point)

    return {
        "single_found": s["found"],
        "planner_found": p["found"],
        "single_args": s["args"],
        "planner_args": p["args"],
        "arg_mismatch": s["args"] != p["args"],
        "single_return": s["returns"],
        "planner_return": p["returns"],
    }


def summarize_case(task_id, single_row, repair_row, planner_row):
    entry_point = planner_row["entry_point"]
    planner_code = planner_row.get("generated_code", "")
    planner_output = planner_row.get("planner_output", "")
    single_code = single_row.get("generated_code", "")

    sig_info = compare_signature(single_code, planner_code, entry_point)
    undefined_names = infer_undefined_names(planner_code, entry_point)
    suspicious_planner_names = planner_mentions_suspicious_names(
        planner_output, planner_code, entry_point
    )

    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "single_status": single_row["status"],
        "repair_status": repair_row["status"],
        "planner_status": planner_row["status"],
        "planner_error_type": planner_row["error_type"],
        "planner_error_message": planner_row["error_message"],
        "signature_arg_mismatch": sig_info["arg_mismatch"],
        "single_args": sig_info["single_args"],
        "planner_args": sig_info["planner_args"],
        "undefined_names": undefined_names,
        "planner_suspicious_names": suspicious_planner_names,
        "planner_output_head": planner_output[:500],
        "planner_code_head": planner_code[:500],
    }


def print_case(case):
    print("=" * 100)
    print(f"TASK: {case['task_id']}")
    print(f"entry_point: {case['entry_point']}")
    print(
        f"single={case['single_status']} | "
        f"repair={case['repair_status']} | "
        f"planner={case['planner_status']} ({case['planner_error_type']})"
    )
    print(f"signature_arg_mismatch: {case['signature_arg_mismatch']}")
    print(f"single_args: {case['single_args']}")
    print(f"planner_args: {case['planner_args']}")
    print(f"undefined_names: {case['undefined_names']}")
    print(f"planner_suspicious_names: {case['planner_suspicious_names']}")
    print("\n[PLANNER OUTPUT HEAD]")
    print(case["planner_output_head"])
    print("\n[PLANNER GENERATED CODE HEAD]")
    print(case["planner_code_head"])
    print("\n[PLANNER ERROR MESSAGE]")
    print(case["planner_error_message"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("single_path", type=str)
    parser.add_argument("repair_path", type=str)
    parser.add_argument("planner_path", type=str)
    args = parser.parse_args()

    single_rows = load_jsonl(args.single_path)
    repair_rows = load_jsonl(args.repair_path)
    planner_rows = load_jsonl(args.planner_path)

    single_map = build_single_map(single_rows)
    repair_map = build_repair_final_map(repair_rows)
    planner_map = build_planner_map(planner_rows)

    # Group A: single=pass, planner=fail(name_error)
    group_a = []
    for task_id, s in single_map.items():
        p = planner_map.get(task_id)
        r = repair_map.get(task_id)
        if p is None or r is None:
            continue

        if s["status"] == "pass" and p["status"] == "fail" and p["error_type"] == "name_error":
            group_a.append(summarize_case(task_id, s, r, p))

    # Group B: planner-only gain
    group_b = []
    for task_id, s in single_map.items():
        p = planner_map.get(task_id)
        r = repair_map.get(task_id)
        if p is None or r is None:
            continue

        if s["status"] != "pass" and r["status"] != "pass" and p["status"] == "pass":
            group_b.append(summarize_case(task_id, s, r, p))

    print("=" * 60)
    print("Group A: single=pass but planner-coder=fail(name_error)")
    print(f"Count: {len(group_a)}")

    mismatch_a = sum(1 for x in group_a if x["signature_arg_mismatch"])
    undefined_a = sum(1 for x in group_a if len(x["undefined_names"]) > 0)
    suspicious_a = sum(1 for x in group_a if len(x["planner_suspicious_names"]) > 0)

    print(f"signature mismatch cases: {mismatch_a}")
    print(f"undefined name cases: {undefined_a}")
    print(f"planner suspicious-name cases: {suspicious_a}")

    print("\nSample cases from Group A")
    for case in group_a[:5]:
        print_case(case)

    print("\n" + "=" * 60)
    print("Group B: planner-only gain")
    print(f"Count: {len(group_b)}")

    mismatch_b = sum(1 for x in group_b if x["signature_arg_mismatch"])
    undefined_b = sum(1 for x in group_b if len(x["undefined_names"]) > 0)
    suspicious_b = sum(1 for x in group_b if len(x["planner_suspicious_names"]) > 0)

    print(f"signature mismatch cases: {mismatch_b}")
    print(f"undefined name cases: {undefined_b}")
    print(f"planner suspicious-name cases: {suspicious_b}")

    print("\nSample cases from Group B")
    for case in group_b[:5]:
        print_case(case)


if __name__ == "__main__":
    main()