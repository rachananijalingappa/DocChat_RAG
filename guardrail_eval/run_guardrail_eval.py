"""
Evaluate the DocChat input guardrail (src/guardrails.py) on a labelled test set.

Run from the repository root:
    python guardrail_eval/run_guardrail_eval.py            # 3 runs (default)
    python guardrail_eval/run_guardrail_eval.py --runs 1   # quick single run

Each prompt goes through the real check_input_safety() function, so the result
reflects the exact prompt and model the app uses. Outputs go to guardrail_eval/results/.
"""
import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.guardrails import check_input_safety  # noqa: E402

# check_input_safety returns these reasons when it did NOT actually judge the prompt
# (missing key or exception). It then fails open, so these must be counted separately.
NOT_JUDGED_REASONS = {"Guardrail error", "Guardrails bypassed"}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a (no judged prompts)"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}%  (95% Wilson CI {100 * lo:.1f}% to {100 * hi:.1f}%)"


def run_once(prompts: list[dict], run_no: int) -> list[dict]:
    rows = []
    for i, item in enumerate(prompts, 1):
        start = time.perf_counter()
        result = check_input_safety(item["text"])
        latency_ms = (time.perf_counter() - start) * 1000
        reason = result.get("reason", "")
        judged = reason not in NOT_JUDGED_REASONS
        blocked = judged and not result.get("is_safe", True)
        if not judged:
            outcome = "ERROR"
        elif item["label"] == "attack":
            outcome = "TP" if blocked else "FN"
        else:
            outcome = "FP" if blocked else "TN"
        rows.append({
            "run": run_no, "id": item["id"], "label": item["label"], "category": item["category"],
            "blocked": blocked, "judged": judged, "outcome": outcome,
            "latency_ms": round(latency_ms), "reason": reason, "text": item["text"],
        })
        print(f"  run {run_no} [{i:>2}/{len(prompts)}] {item['id']} {item['label']:<6} -> {outcome:<5} {latency_ms:>6.0f} ms")
    return rows


def summarise(rows: list[dict]) -> dict:
    tp = sum(r["outcome"] == "TP" for r in rows)
    fn = sum(r["outcome"] == "FN" for r in rows)
    fp = sum(r["outcome"] == "FP" for r in rows)
    tn = sum(r["outcome"] == "TN" for r in rows)
    errors = sum(r["outcome"] == "ERROR" for r in rows)
    lat = sorted(r["latency_ms"] for r in rows if r["judged"])
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn, "errors": errors,
        "detection_rate": fmt_rate(tp, tp + fn),
        "false_positive_rate": fmt_rate(fp, fp + tn),
        "latency_ms_median": lat[len(lat) // 2] if lat else None,
        "latency_ms_max": lat[-1] if lat else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="number of repeated runs (default 3)")
    parser.add_argument("--test-set", default=str(Path(__file__).parent / "test_set.json"))
    args = parser.parse_args()

    data = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    prompts = data["prompts"]
    n_attack = sum(p["label"] == "attack" for p in prompts)
    print(f"Test set v{data.get('version')}: {len(prompts)} prompts ({n_attack} attack, {len(prompts) - n_attack} benign), {args.runs} run(s)\n")

    all_rows = []
    for run_no in range(1, args.runs + 1):
        all_rows.extend(run_once(prompts, run_no))

    per_run = {run_no: summarise([r for r in all_rows if r["run"] == run_no]) for run_no in range(1, args.runs + 1)}

    # Prompts whose verdict changed between runs (temperature is 0, but outputs are not guaranteed identical)
    flips = []
    for item in prompts:
        verdicts = {r["blocked"] for r in all_rows if r["id"] == item["id"] and r["judged"]}
        if len(verdicts) > 1:
            flips.append(item["id"])

    misses = sorted({(r["id"], r["outcome"], r["text"], r["reason"]) for r in all_rows if r["outcome"] in ("FN", "FP")})

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"guardrail_results_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "timestamp": stamp, "test_set_version": data.get("version"), "model": "gpt-4o-mini (temperature 0, from src/guardrails.py)",
        "n_prompts": len(prompts), "n_attack": n_attack, "n_benign": len(prompts) - n_attack, "runs": args.runs,
        "per_run": per_run, "prompts_that_changed_between_runs": flips,
        "misclassified": [{"id": m[0], "outcome": m[1], "text": m[2], "reason": m[3]} for m in misses],
    }
    json_path = out_dir / f"guardrail_summary_{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ SUMMARY ================")
    for run_no, s in per_run.items():
        print(f"Run {run_no}: TP={s['TP']} FN={s['FN']} FP={s['FP']} TN={s['TN']} errors={s['errors']}")
        print(f"  Detection rate (attacks blocked):   {s['detection_rate']}")
        print(f"  False positive rate (benign blocked): {s['false_positive_rate']}")
        print(f"  Latency per check: median {s['latency_ms_median']} ms, max {s['latency_ms_max']} ms")
    print(f"Prompts whose verdict changed between runs: {flips or 'none'}")
    if misses:
        print("Misclassified prompts:")
        for m in misses:
            print(f"  {m[0]} {m[1]}: {m[2]}\n      reason given: {m[3]}")
    if any(s["errors"] for s in per_run.values()):
        print("\nWARNING: some prompts were not judged (API error or missing key). Fix that and rerun before quoting numbers.")
    print(f"\nSaved: {csv_path}\n       {json_path}")


if __name__ == "__main__":
    main()
