"""Repeat the FP32 feature matrix in upstream/native/native/upstream order."""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np

from .common import save_json
from .features_report import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["laya", "laya-multilingual", "laya-typed-decisions"]
    )
    parser.add_argument("--backend", choices=["coreml", "mlx"], default="coreml")
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    mlx = Path(__file__).resolve().parents[1]
    core = mlx.parent / "laya-coreml"
    torch_version = subprocess.check_output(
        [str(mlx / ".venv/bin/python"), "-c", "import torch; print(torch.__version__)"], text=True
    ).strip()
    assert torch_version == "2.14.0", "Use the qualified upstream reference environment"
    args.directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        reports = {"upstream": [], args.backend: []}
        for index, backend in enumerate(["upstream", args.backend, args.backend, "upstream"]):
            output = args.directory.resolve() / f"{model}-{index}-{backend}.json"
            python = (core if backend == "coreml" else mlx) / ".venv/bin/python"
            cmd = [
                str(python),
                "-m",
                "benchmarks.features",
                "--backend",
                backend,
                "--model",
                model,
                "--precision",
                "float32",
                "--workloads",
                "single",
                "batch8",
                "long",
                "structured",
                "stability",
                "mixed_shapes",
                "--warmup",
                "5",
                "--iterations",
                "30",
                "--output",
                str(output),
            ]
            if backend == "coreml":
                cmd += ["--coreml-model", str(core / f"models/v0412/{model}-fp32-dense-adaptive")]
            subprocess.run(cmd, cwd=mlx, check=True)
            reports[backend].append(json.loads(output.read_text()))
        reference = reports["upstream"][0]
        for i, ref in enumerate(reference["results"]):
            samples = {}
            medians = {}
            maximum = 0.0
            for backend, runs in reports.items():
                samples[backend] = []
                medians[backend] = []
                for run in runs:
                    row = run["results"][i]
                    assert row["workload"] == ref["workload"]
                    assert row["input_sha256"] == ref["input_sha256"]
                    assert run["source_weights_sha256"] == reference["source_weights_sha256"]
                    errors, differences = compare(ref["output"], row["output"])
                    maximum = max(maximum, max(errors, default=0))
                    assert not differences and maximum <= 1e-4, (
                        model,
                        row["workload"],
                        differences,
                        maximum,
                    )
                    samples[backend].extend(row["timing"]["samples_ms"])
                    medians[backend].append(row["timing"]["p50_ms"])
            rows.append(
                {
                    "model": model,
                    "workload": ref["workload"],
                    "backend": args.backend,
                    "run_p50_ms": medians,
                    "pooled_p50_ms": {k: statistics.median(v) for k, v in samples.items()},
                    "pooled_p95_ms": {k: float(np.percentile(v, 95)) for k, v in samples.items()},
                    "pooled_mean_ms": {k: statistics.mean(v) for k, v in samples.items()},
                    "p50_speedup": statistics.median(samples["upstream"])
                    / statistics.median(samples[args.backend]),
                    "throughput_ratio": statistics.mean(samples["upstream"])
                    / statistics.mean(samples[args.backend]),
                    "max_numeric_error": maximum,
                    "parity_passed": True,
                }
            )
        save_json(
            args.directory / "paired-summary.json",
            {
                "order": ["upstream", args.backend, args.backend, "upstream"],
                "torch": torch_version,
                "precision": "float32",
                "warmup": 5,
                "timed_calls_per_process_workload": 30,
                "rows": rows,
                "note": "All runs retained; pooled medians and mean-based throughput ratios use both runs. Shared desktop load remains uncontrolled. Model loading and warmup excluded.",
            },
        )
    print("PAIRED CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
