#!/usr/bin/env python3
"""Summarize hardening cases: metric changes versus the reference run.

score change = sum_experiment weight * sum_row sum_metric |coefficient| *
(improvement in the metric's better direction). Every row present in the
reference scores is matched, so changes to comparison rows would show up.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]


def rows(scores):
    return {(exp, row): metrics for exp, block in scores["experiments"].items()
            for row, metrics in block["results"].items()}


def change(case_scores, ref_scores):
    total, per_metric, per_row = 0.0, {}, {}
    ref_rows = rows(ref_scores)
    for (exp, row), metrics in rows(case_scores).items():
        if (exp, row) not in ref_rows or exp not in spec:
            continue
        cfg = spec[exp]
        row_total = 0.0
        for m, value in metrics.items():
            if m not in cfg["metrics"]:
                continue
            sign = 1.0 if cfg["metrics"][m]["higher_is_better"] else -1.0
            term = cfg["weight"] * abs(cfg["metrics"][m]["coefficient"]) * sign * (value - ref_rows[(exp, row)][m])
            per_metric[f"{exp}/{row}/{m}"] = term
            row_total += term
        per_row[f"{exp}/{row}"] = row_total
        total += row_total
    return total, per_row, per_metric


def main():
    ref = json.loads((HERE / "reference/scores_test.json").read_text())
    summary = {}
    for folder in sorted(p for p in HERE.iterdir() if p.is_dir()):
        path = folder / "scores_test.json"
        if not path.exists():
            summary[folder.name] = {"error": (folder / "error.txt").read_text() if (folder / "error.txt").exists() else "no scores"}
            continue
        scores = json.loads(path.read_text())
        total, per_row, per_metric = change(scores, ref)
        proposed = {exp: {row: m for row, m in block["results"].items() if row.startswith("generator_")}
                    for exp, block in scores["experiments"].items()}
        raw = {row: v for row, v in scores.get("_raw", {}).items() if row.startswith("generator_")}
        failures = {row: v for row, v in scores.get("_failures", {}).items()}
        summary[folder.name] = {"score_change": total, "row_changes": per_row, "proposed": proposed,
                                "raw": raw, "failures": failures}
        if folder.name == "reference":
            summary[folder.name]["comparators_raw"] = {r: v for r, v in scores.get("_raw", {}).items()
                                                      if not r.startswith("generator_")}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    for name, item in summary.items():
        if "score_change" not in item:
            print(f"{name:32s} ERROR {item['error'][:80]}")
            continue
        p = item["proposed"]
        cells = []
        for exp, row in (("codec_stack2", "generator_2x"), ("codec_stack4", "generator_4x")):
            m = p.get(exp, {}).get(row)
            if m:
                cells.append(f"{row[-2:]}: val={m['dependency_validity']:.4f} fd={m['codec_fd']:.4f} sfd={m['speaker_conditioned_codec_fd']:.3f}")
        fail = ",".join(sorted(r for r in item["failures"] if r.startswith("generator_")))
        print(f"{name:32s} change={item['score_change']:+.5f} | " + " | ".join(cells) + (f" | FAILED {fail}" if fail else ""))


if __name__ == "__main__":
    main()
