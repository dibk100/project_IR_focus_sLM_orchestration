# project_IR_focus_sLM_orchestration/scripts/make_repeat_config.py
import argparse
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True, help="Path to the base YAML config")
    parser.add_argument("--seed", type=int, required=True, help="Fixed seed value")
    parser.add_argument("--repeat", type=int, required=True, help="Repeat index")
    parser.add_argument("--output", required=True, help="Path to save the generated YAML config")
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("run", {})
    cfg.setdefault("output", {})

    base_run_id = cfg["run"].get("run_id", "policy_run")
    base_output_dir = cfg["output"].get("dir", "results/policy_run")

    cfg["run"]["seed"] = args.seed
    cfg["run"]["run_id"] = f"{base_run_id}_seed{args.seed}_repeat{args.repeat}"
    cfg["output"]["dir"] = f"{base_output_dir}_seed{args.seed}_repeat{args.repeat}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()