"""Exercise real native logits through the pinned official calibration contract.

Targets are deterministic synthetic labels, not a quality evaluation dataset.
Repeated records cross the official per-bucket fitting/holdout floors without
lowering them. No checkpoint config or weights are written.
"""

import argparse
import copy
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .common import ROOT, UPSTREAM_REVISION, digest, environment, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "coreml"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / ".upstream"), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == UPSTREAM_REVISION
    sys.path.insert(0, str(ROOT / ".upstream"))
    official = importlib.import_module("laya.calibrate")
    sys.path.insert(0, str(ROOT.parent / "laya-coreml"))
    package = importlib.import_module("laya_" + args.backend)
    common = importlib.import_module("laya_" + args.backend + ".common")
    import torch

    torch.set_num_threads(1)
    fixture = json.loads(
        (ROOT.parent / "laya-coreml/benchmarks/results/reference-v0412.json").read_text()
    )
    native_hash = hashlib.sha256()
    source = Path(package.__file__).parent
    for path in sorted(source.rglob("*.py")):
        native_hash.update(str(path.relative_to(source)).encode())
        native_hash.update(path.read_bytes())
    report = {
        "backend": args.backend,
        "native_source_sha256": native_hash.hexdigest(),
        "reference_sha256": hashlib.sha256(
            (ROOT.parent / "laya-coreml/benchmarks/results/reference-v0412.json").read_bytes()
        ).hexdigest(),
        "environment": environment(),
        "upstream_revision": commit,
        "target_source": "synthetic reference-argmax labels with deterministic perturbations; compatibility only",
        "models": [],
    }
    for name in ("laya", "laya-multilingual", "laya-typed-decisions"):
        folder = (
            ROOT / "models" / name
            if args.backend == "mlx"
            else ROOT.parent
            / "laya-coreml/models/v0412"
            / ("laya-adaptive-v2" if name == "laya" else name + "-adaptive")
        )
        with package.Agent(folder) as agent:
            original_config = copy.deepcopy(agent.cfg)
            records = []
            cases = fixture["models"][name]["cases"]
            for case in cases:
                items, _ = agent.prepare(case["state"], case["questions"])
                batch = common.collate_items([items], agent.tok.pad_token_id)
                logits, _ = agent._forward(batch)
                for i, item in enumerate(items):
                    k = len(item["markers"])
                    target = np.zeros(k, np.float32)
                    chosen = int(np.argmax(case["logits"][i][:k]))
                    if len(records) % 5 == 0:
                        chosen = (chosen + 1) % k
                    target[chosen] = 1
                    records.append((item["qtype"], logits[i, :k].copy(), target, k))
            # Keep upstream fitting floors; repeat data solely to exercise branches.
            expanded = records * 320
            expected = official.fit_temperature_map(expanded, compute_ece=True, seed=17)
            actual = agent.fit_temperatures(expanded, compute_ece=True, seed=17)
            assert actual == expected, (
                "Native fitting differs from official on identical real logits"
            )
            assert actual["temperature_by_options"]
            assert actual["report"]["n_eval"] > 0
            expected_bins = official.fit_binning_map(
                expanded, actual["temperature"], actual["temperature_by_options"]
            )
            actual_bins = agent.fit_binning(expanded)
            assert actual_bins == expected_bins and actual_bins
            case = cases[0]
            calibrated = agent.predict(case["state"], case["questions"])
            with tempfile.TemporaryDirectory(prefix="laya-calibration-contract-") as directory:
                path = Path(directory) / "calibration.json"
                agent.save_calibration(str(path))
                payload = json.loads(path.read_text())
                agent.temperature = [1.0, 1.0, 1.0]
                agent.temperature_by_options = {}
                agent.binning_map = None
                agent.load_calibration(str(path))
                assert agent.predict(case["state"], case["questions"]) == calibrated
                assert agent.temperature == actual["temperature"]
                assert agent.temperature_by_options == actual["temperature_by_options"]
                assert agent.binning_map == actual_bins
            assert agent.cfg == original_config
            report["models"].append(
                {
                    "model": name,
                    "unique_forward_records": len(records),
                    "records_sha256": digest(
                        [(int(qt), z.tolist(), target.tolist(), k) for qt, z, target, k in records]
                    ),
                    "expanded_records": len(expanded),
                    "fit_exactly_matches_official": True,
                    "binning_exactly_matches_official": True,
                    "roundtrip_preserves_prediction": True,
                    "checkpoint_config_unchanged": True,
                    "fit": actual,
                    "binning": actual_bins,
                    "calibration_payload": payload,
                    "prediction_sha256": digest(calibrated),
                }
            )
            save_json(args.output, report)
            print(args.backend, name, "calibration differential and roundtrip passed", flush=True)
    report["passed"] = True
    save_json(args.output, report)


if __name__ == "__main__":
    main()
