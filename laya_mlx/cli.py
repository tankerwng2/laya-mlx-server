"""Command-line prediction and checkpoint conversion."""

import argparse
import json
from pathlib import Path

from . import __version__


def _read_json(path, option):
    """Name the input file on decoding/read errors before loading a checkpoint."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(
            "%s file %s cannot be read as UTF-8 JSON: %s" % (option, path, error)
        ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-mlx", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "convert"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", default="convaiinnovations/laya")
        sub.add_argument("--subfolder")
        sub.add_argument("--revision")
        sub.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
        if command == "predict":
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument("--state", help="Plain text input")
            source.add_argument("--state-file", type=Path, help="JSON state file")
            sub.add_argument(
                "--questions", required=True, type=Path, help="JSON question definitions"
            )
            sub.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
            sub.add_argument("--batch-size", type=int, default=16)
        else:
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "convert":
        from .convert import convert

        result = convert(
            args.model,
            args.output,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
        )
        print(json.dumps({"output": str(result), "dtype": args.dtype}))
    else:
        try:
            state = (
                args.state
                if args.state is not None
                else _read_json(args.state_file, "--state-file")
            )
            questions = _read_json(args.questions, "--questions")
        except ValueError as error:
            parser.error(str(error))
        from .agent import Agent

        agent = Agent(
            args.model,
            device=args.device,
            dtype=args.dtype,
            revision=args.revision,
            subfolder=args.subfolder,
            batch_size=args.batch_size,
        )
        print(json.dumps(agent.predict(state, questions), ensure_ascii=False, indent=2))
