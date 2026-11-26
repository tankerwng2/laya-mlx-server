"""Sequential final v0.4.12 correctness/performance campaign for sibling checkouts.

Requires exported B=1/2/4/16 dense FP32 and FP16 adaptive bundles. Uses each repository's development environment; no GPU jobs
should run concurrently. Reports retain source/checkpoint hashes and precision.
"""

import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    core = root / "laya-coreml"
    mlx = root / "laya-mlx"
    models = ("laya", "laya-multilingual", "laya-typed-decisions")
    reference = json.loads(
        (core / "benchmarks/results/reference-v0412-fp32-torch214.json").read_text()
    )
    torch_version = subprocess.check_output(
        [
            str(mlx / ".venv/bin/python"),
            "-c",
            "import importlib.metadata as m; print(m.version('torch'))",
        ],
        text=True,
    ).strip()
    assert torch_version == reference["versions"]["torch"], "Reference runtime version mismatch"
    for name in models:
        for precision in ("fp32", "fp16"):
            assert (
                core / f"models/v0412/{name}-{precision}-dense-adaptive/coreml_config.json"
            ).exists()
    for name in models:
        for precision in ("float32", "float16"):
            suffix = "fp32-dense-adaptive" if precision == "float32" else "fp16-dense-adaptive"
            subprocess.run(
                [
                    str(core / ".venv/bin/python"),
                    "-m",
                    "benchmarks.validate",
                    f"models/v0412/{name}-{suffix}",
                    "--name",
                    name,
                    "--reference",
                    "benchmarks/results/reference-v0412-fp32-torch214.json",
                    "--repeats",
                    "100",
                    "--output",
                    f"benchmarks/results/v0412-{name}-{precision}-cached-validation.json",
                ],
                cwd=core,
                check=True,
            )
    for name in models:
        for backend in ("upstream", "mlx", "coreml"):
            python = (core if backend == "coreml" else mlx) / ".venv/bin/python"
            cmd = [
                str(python),
                "-m",
                "benchmarks.features",
                "--backend",
                backend,
                "--model",
                name,
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
                f"benchmarks/results/v0412-final/{name}-{backend}-float32.json",
            ]
            if backend == "coreml":
                cmd += ["--coreml-model", str(core / f"models/v0412/{name}-fp32-dense-adaptive")]
            subprocess.run(cmd, cwd=mlx, check=True)
    subprocess.run(
        [
            str(mlx / ".venv/bin/python"),
            "-m",
            "benchmarks.features_report",
            "--directory",
            "benchmarks/results/v0412-final",
        ],
        cwd=mlx,
        check=True,
    )
    summary = json.loads((mlx / "benchmarks/results/v0412-final/summary.json").read_text())
    assert all(row["parity_passed"] for row in summary["rows"]), "Public output mismatch"
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
