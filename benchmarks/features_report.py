"""Summarize native feature timing and compare full outputs to upstream FP32."""

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path

from .common import save_json


def compare(expected, actual, path=""):
    errors, differences = [], []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            differences.append(path + ": keys differ")
        for key in expected.keys() & actual.keys():
            e, d = compare(expected[key], actual[key], path + "/" + key)
            errors.extend(e)
            differences.extend(d)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(path + ": lengths differ")
        for i, (left, right) in enumerate(zip(expected, actual)):
            e, d = compare(left, right, path + "/" + str(i))
            errors.extend(e)
            differences.extend(d)
    elif isinstance(expected, float) and type(actual) in (float, int):
        if not math.isfinite(expected) or not math.isfinite(actual):
            differences.append(path + ": non-finite public value")
        else:
            # Compare the serialized decimal values. A one-step 0.0001 change
            # must not fail because binary subtraction yields 0.00010000000000000286.
            errors.append(float(abs(Decimal(str(expected)) - Decimal(str(actual)))))
    elif type(expected) is not type(actual) or expected != actual:
        differences.append(f"{path}: {expected!r} != {actual!r}")
    return errors, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmarks/results/v0412-features"))
    args = parser.parse_args()
    rows = []
    for file in sorted(args.directory.glob("*.json")):
        if file.name == "summary.json":
            continue
        report = json.loads(file.read_text())
        reference = json.loads(
            (args.directory / (report["model"] + "-upstream-float32.json")).read_text()
        )
        assert report["source_weights_sha256"] == reference["source_weights_sha256"]
        assert (
            report["environment"]["upstream_revision"]
            == reference["environment"]["upstream_revision"]
        )
        base = {row["workload"]: row for row in reference["results"]}
        for row in report["results"]:
            ref = base[row["workload"]]
            assert ref["input_sha256"] == row["input_sha256"]
            errors, differences = compare(ref["output"], row["output"])
            maximum = max(errors, default=0)
            precision = report["precision"]
            full_fp32 = (
                precision.get("compute") == "float32"
                or precision.get("export_compute") == "float32"
                or precision.get("requested") == "float32"
            )
            tolerance = 1e-4 if full_fp32 else 0.02
            rows.append(
                {
                    "file": file.name,
                    "model": report["model"],
                    "backend": report["backend"],
                    "precision": report["precision"],
                    "workload": row["workload"],
                    "p50_ms": row["timing"]["p50_ms"],
                    "p95_ms": row["timing"]["p95_ms"],
                    "mean_ms": row["timing"]["mean_ms"],
                    "decisions_per_second": row["timing"]["decisions_per_second"],
                    "process_peak_rss_bytes": row["process_peak_rss_bytes"],
                    "upstream_torch_version": reference["environment"]["packages"]["torch"],
                    "speedup_vs_upstream_fp32": ref["timing"]["p50_ms"] / row["timing"]["p50_ms"],
                    "throughput_ratio_vs_upstream_fp32": row["timing"]["decisions_per_second"]
                    / ref["timing"]["decisions_per_second"],
                    "max_numeric_error": maximum,
                    "semantic_differences": differences,
                    "float_tolerance": tolerance,
                    "parity_passed": not differences and maximum <= tolerance,
                    "parity_passed_at_0_02": not differences and maximum <= 0.02,
                }
            )
    save_json(
        args.directory / "summary.json",
        {
            "rows": rows,
            "note": "FP16 upstream is autocast with FP32 weights; native FP16 is not identical arithmetic. "
            "Use FP32/FP32 for matched precision. Core ML partial runs remain partial. "
            "Full nested outputs compared: categories, structure and integer diagnostics exact, floats <=0.0001 for FP32 and <=0.02 for reduced precision. "
            "This finite workload suite does not establish universal numerical equivalence.",
        },
    )
    for row in rows:
        print(
            row["file"],
            row["workload"],
            round(row["p50_ms"], 2),
            round(row["speedup_vs_upstream_fp32"], 2),
            row["parity_passed"],
        )


if __name__ == "__main__":
    main()
