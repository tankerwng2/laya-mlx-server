"""Run the pinned upstream SDK's unchanged live HTTP assertions on native inference.

Only the server fixture's checkpoint construction is replaced with a local native
checkpoint. Router attachment, requests, auth/limits, strict projection and tournament
assertions remain the upstream implementation.
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = "laya_mlx" if (root / "laya_mlx").exists() else "laya_coreml"
    upstream = root / ".upstream"
    source = (upstream / "tests/sdk_server_fixture.py").read_text()
    start = source.index("    torch.manual_seed(7)")
    end = source.index("        questions = {", start)
    # Keep the fixture's indentation and all of its routing/server setup intact.
    source = (
        source[:start]
        + (f"    with load({str(args.model.resolve())!r}) as agent:\n")
        + source[end:]
    )
    source = "\n".join(
        line
        for line in source.splitlines()
        if not (
            line.startswith("import torch")
            or line.startswith("from safetensors.")
            or line.startswith("from tokenizers")
            or line.startswith("from transformers")
            or line.startswith("from laya.common")
            or line.startswith("sys.path.insert")
        )
    )
    source = source.replace("from laya import", f"from {package} import").replace(
        "from laya.serve import", f"from {package}.serve import"
    )
    source = source.replace("import torch  # noqa: E402", "")
    source = source.replace("import uvicorn", f"sys.path.insert(0, {str(root)!r})\nimport uvicorn")
    with tempfile.TemporaryDirectory(prefix="laya-native-sdk-") as directory:
        fixture = Path(directory) / "fixture.py"
        fixture.write_text(source)
        # The official harness resolves its interpreter path; preserve the venv
        # through a real launcher file instead of resolving the python symlink.
        launcher = Path(directory) / "server-python"
        launcher.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
        launcher.chmod(0o755)
        env = {**os.environ, "PYTHONPATH": str(upstream / "sdk/python/src")}
        subprocess.run(
            [
                sys.executable,
                str(upstream / "sdk/python/tests/integration.py"),
                "--server-python",
                str(launcher),
                "--server-fixture",
                str(fixture),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
