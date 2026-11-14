"""Exercise the installed inference-only wheel offline, outside the source checkout."""

import argparse
import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import laya_mlx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()
    assert Path(laya_mlx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    absent = {name: importlib.util.find_spec(name) is None for name in ("torch", "transformers")}
    assert all(absent.values()), absent
    attempts = []
    original_connect = socket.socket.connect

    def offline(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("Unexpected network access during local wheel inference")

    source = Path(laya_mlx.__file__).parent
    source_hash = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        source_hash.update(str(path.relative_to(source)).encode())
        source_hash.update(path.read_bytes())
    report = {
        "native_source_sha256": source_hash.hexdigest(),
        "package": laya_mlx.__version__,
        "module": laya_mlx.__file__,
        "absent_dependencies": absent,
        "device": args.device,
        "models": [],
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": ["billing", "technical", "other"],
        }
    }
    state = "The customer requests a refund for a duplicate charge."
    schema = {
        "type": "object",
        "properties": {"department": {"enum": ["billing", "technical", "other"]}},
    }
    socket.socket.connect = offline
    try:
        for name in ("laya", "laya-multilingual", "laya-typed-decisions"):
            with laya_mlx.Agent(
                args.model_root / name, device=args.device, cache_prompts=True
            ) as agent:
                warmup_seconds = agent.warmup()
                assert warmup_seconds >= 0
                single = agent.predict(state, questions)
                repeats = [agent.predict(state, questions) for _ in range(3)]
                assert all(value == single for value in repeats)
                batch = agent.predict_batch([state, state], questions, batch_size=2)
                assert len(batch) == 2
                assert all(
                    value["answers"]["department"]["choice"]
                    == single["answers"]["department"]["choice"]
                    for value in batch
                )
                scanned = agent.predict_long(state * 8, questions, window=64, stride=32)
                assert scanned["usage"]["windows"] > 1
                structured = agent.decide(state, schema)
                assert structured["department"] in schema["properties"]["department"]["enum"]
                hook = laya_mlx.OptionStabilityHook(
                    accept=None, escalate=None, variants="fast", seed="smoke"
                )
                probed = agent.predict(state, questions, hooks=[hook])
                assert "reliability" in probed["answers"]["department"]
                report["models"].append(
                    {
                        "model": name,
                        "single": single,
                        "batch": batch,
                        "long": scanned,
                        "structured": structured,
                        "stability": probed,
                        "repeat_calls": 3,
                        "warmup_seconds": warmup_seconds,
                        "passed": True,
                    }
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(name, "passed", flush=True)
        with laya_mlx.Router(
            models={"english": str(args.model_root / "laya")}, local_files_only=True
        ) as router:
            routed_agent = router.load("english")
            assert routed_agent.predict(state, questions)["answers"]["department"]["choice"] in (
                "billing",
                "technical",
                "other",
            )
        report["offline_router"] = True
    finally:
        socket.socket.connect = original_connect
    assert not attempts
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    report["network_attempts"] = len(attempts)
    report["passed"] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
