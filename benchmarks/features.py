"""Isolated end-to-end migration workloads; includes outputs for differential review.

Run each backend sequentially, never concurrently. FP16 upstream uses MPS autocast;
native FP16 weights and Core ML export precision are distinct and reported explicitly.
"""

import argparse
import functools
import hashlib
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import MODELS, ROOT, digest, environment, load_reference, save_json
from .worker import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["upstream", "mlx", "coreml"], required=True)
    parser.add_argument("--model", default="laya")
    parser.add_argument("--coreml-model", type=Path)
    parser.add_argument("--fast-prediction", action="store_true")
    parser.add_argument("--frequent-reshape", action="store_true")
    parser.add_argument("--shape-cache-experiment", action="store_true")
    parser.add_argument("--shape-cache-size", type=int, default=4)
    parser.add_argument(
        "--compute-units", default="cpu_gpu", choices=["cpu", "cpu_gpu", "cpu_ne", "all"]
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["single", "batch8", "long", "structured", "stability", "mixed_shapes"],
    )
    parser.add_argument("--precision", choices=["float32", "float16", "auto"], required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations and warmup must be positive")
    if args.backend == "upstream":
        # Public upstream environment switch, set before constructing the agent.
        if args.precision != "auto":
            os.environ["LAYA_MPS_AMP_MIN_ROWS"] = "1" if args.precision == "float16" else "1000000"
        agent = load_reference(ROOT / "models" / args.model)
        if args.precision == "float32":
            agent.amp_enabled = False
        import laya as package
        import torch

        synchronize = torch.mps.synchronize
        precision = {
            "requested": args.precision,
            "weights": "float32",
            "autocast": str(agent.dtype),
            "amp_min_rows": agent.mps_amp_min_rows,
            "amp_enabled": agent.amp_enabled,
        }
    elif args.backend == "mlx":
        if args.precision == "auto":
            parser.error("MLX requires explicit precision")
        import mlx.core as mx

        import laya_mlx as package

        agent = package.Agent(ROOT / "models" / args.model, dtype=args.precision)
        synchronize = mx.synchronize
        precision = {"weights": args.precision, "compute": args.precision}
    else:
        if args.precision == "auto":
            parser.error("Core ML requires explicit export precision")
        sys.path.insert(0, str(ROOT.parent / "laya-coreml"))
        import laya_coreml as package

        if args.fast_prediction or args.frequent_reshape:
            import coremltools as ct

            # Process-local experiment; no production default changes.
            hints = {}
            if args.fast_prediction:
                hints["specializationStrategy"] = ct.SpecializationStrategy.FastPrediction
            if args.frequent_reshape:
                hints["reshapeFrequency"] = ct.ReshapeFrequency.Frequent
            ct.models.MLModel = functools.partial(ct.models.MLModel, optimization_hints=hints)

        agent = package.Agent(
            args.coreml_model or ROOT.parent / "laya-coreml/models/hub" / (args.model + "-coreml"),
            compute_units=args.compute_units,
            shape_cache_size=args.shape_cache_size,
        )

        if args.shape_cache_experiment:
            from collections import OrderedDict

            import coremltools as ct

            original = agent._graph_outputs
            clones = OrderedDict()

            def cached(model, batch):
                key = (id(model), batch["input_ids"].shape)
                if key not in clones:
                    if len(clones) >= 4:
                        clones.popitem(last=False)
                    clones[key] = ct.models.MLModel(
                        model.package_path, compute_units=model.compute_unit
                    )
                clones.move_to_end(key)
                return original(clones[key], batch)

            agent._graph_outputs = cached

        def synchronize():
            pass  # Core ML predict is synchronous

        precision = {
            "export_compute": agent.manifest["precision"],
            "compute_units": agent.compute_units,
            "shape": agent.shape,
            "fast_prediction": args.fast_prediction,
            "frequent_reshape": args.frequent_reshape,
            "shape_cache_experiment": args.shape_cache_experiment,
            "shape_cache_size": getattr(getattr(agent, "_shape_cache", None), "capacity", 0),
        }
    state = "The customer requests a refund for a duplicate charge."
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": ["billing", "technical", "other"],
        },
        "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    }
    states = [state, "My password does not work.", "", state * 4] * 2
    schema = {
        "type": "object",
        "properties": {"department": {"enum": ["billing", "technical", "other"]}},
    }
    hook = package.OptionStabilityHook(accept=None, escalate=None, variants="fast", seed="matrix")
    workloads = [
        ("single", lambda: agent.predict(state, questions), [state, questions], 2),
        (
            "batch8",
            lambda: agent.predict_batch(states, questions, batch_size=8),
            [states, questions],
            16,
        ),
        (
            "long",
            lambda: agent.predict_long(state * 16, questions, window=64, stride=32),
            [state * 16, questions, 64, 32],
            2,
        ),
        ("structured", lambda: agent.decide(state, schema), [state, schema], 1),
        (
            "stability",
            lambda: agent.predict(state, questions, hooks=[hook]),
            [state, questions, "fast", "matrix"],
            2,
        ),
    ]
    if args.workloads and "mixed_shapes" in args.workloads:
        mixed_states = ["", state * 8, state, state * 2]
        workloads.append(
            (
                "mixed_shapes",
                lambda: [agent.predict(value, questions) for value in mixed_states],
                [mixed_states, questions],
                8,
            )
        )
    source = Path(package.__file__).parent
    native_digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        native_digest.update(str(path.relative_to(source)).encode())
        native_digest.update(path.read_bytes())
    manifest = getattr(agent, "manifest", None)
    weights = hashlib.sha256()
    with (ROOT / "models" / args.model / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            weights.update(block)
    weights_hash = weights.hexdigest()
    if manifest and manifest["source_weights_sha256"] != weights_hash:
        raise ValueError("Core ML bundle and upstream checkpoint weights differ")
    if manifest and manifest["precision"] != args.precision:
        raise ValueError("Requested benchmark precision differs from the export")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "environment": environment(),
        "native_source_sha256": native_digest.hexdigest(),
        "source_checkpoint_revision": MODELS[args.model],
        "source_weights_sha256": weights_hash,
        "coreml_manifest": manifest,
        "backend": args.backend,
        "model": args.model,
        "precision": precision,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing": "wall clock, completed end-to-end calls; model load excluded",
        "results": [],
    }
    for name, function, inputs, decisions in workloads:
        if args.workloads and name not in args.workloads:
            continue
        output = function()
        timing = measure(function, synchronize, args.warmup, args.iterations)
        timing["decisions_per_second"] = decisions * 1000 / timing["mean_ms"]
        report["results"].append(
            {
                "workload": name,
                "input_sha256": digest(inputs),
                "output": output,
                "timing": timing,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        save_json(args.output, report)
        print(
            args.backend, args.precision, args.model, name, round(timing["p50_ms"], 3), flush=True
        )


if __name__ == "__main__":
    main()
