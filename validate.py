#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def validate_reference(obj):
    assert "experiments" in obj and isinstance(obj["experiments"], dict)
    total = 0.0
    for name, exp in obj["experiments"].items():
        for key in ["description", "weight", "primary_metric", "metrics", "results"]:
            assert key in exp, f"{name}: missing {key}"
        total += float(exp["weight"])
        assert exp["primary_metric"] in exp["metrics"], f"{name}: primary metric not defined"
        for metric, spec in exp["metrics"].items():
            assert "higher_is_better" in spec, f"{name}/{metric}: missing direction"
            assert "coefficient" in spec, f"{name}/{metric}: missing coefficient"
        for method, metrics in exp["results"].items():
            assert "type" in metrics, f"{name}/{method}: missing type"
            for metric in exp["metrics"]:
                assert metric in metrics, f"{name}/{method}: missing {metric}"
    assert abs(total - 1.0) < 1e-6, f"experiment weights sum to {total}, not 1"


def compare(ref, scores):
    print("experiment,method,metric,reference,score")
    for exp_name, exp in scores.get("experiments", {}).items():
        ref_exp = ref.get("experiments", {}).get(exp_name, {})
        for method, vals in exp.get("results", {}).items():
            ref_vals = ref_exp.get("results", {}).get(method, {})
            for metric, score in vals.items():
                if metric == "type":
                    continue
                print(f"{exp_name},{method},{metric},{ref_vals.get(metric, '')},{score}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="scoring/reference.json")
    parser.add_argument("--scores", default="scoring/scores.json")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    ref = load(args.reference)
    validate_reference(ref)
    print(f"validated {args.reference}")
    if args.compare:
        compare(ref, load(args.scores))


if __name__ == "__main__":
    main()
