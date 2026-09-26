# Portable entry points. No absolute paths, no host assumptions.
# MLX is macOS/arm64 only (see pyproject marker); on Linux use upstream laya-serve.
VENV ?= .venv
BIN  := $(VENV)/bin
PORT ?= 8080
HOST ?= 127.0.0.1
MODELS ?= multilingual
# UV_INDEX is opt-in: set it to a regional PyPI mirror when files.pythonhosted.org
# is slow or unreachable, e.g. make install UV_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
UV_INDEX ?=

.PHONY: lock install prefetch http mcp test lint build clean

lock:
	uv lock

# --all-packages pulls in the server/ workspace member (fastapi, uvicorn).
install:
	$(if $(UV_INDEX),UV_DEFAULT_INDEX=$(UV_INDEX) )uv sync --all-packages --extra dev --extra demo
	# uv's --extra flag only reaches the root project, so the member's mcp extra
	# is installed explicitly into the same venv.
	$(if $(UV_INDEX),UV_DEFAULT_INDEX=$(UV_INDEX) )uv pip install --python $(BIN)/python -e "./server[mcp]"

# One-time weight fetch for the aliases in MODELS (repo ids come from the
# server's MLX_CHECKPOINTS table, so there is nothing to keep in sync).
# Point HF_ENDPOINT at a mirror where huggingface.co is blocked.
prefetch:
	HF_ENDPOINT=$${HF_ENDPOINT:-https://hf-mirror.com} LAYA_MODELS='$(MODELS)' $(BIN)/python -c \
	  'import os; from laya_mlx_server.runtime import MLX_CHECKPOINTS; from huggingface_hub import snapshot_download; wanted = [n.strip() for n in os.environ["LAYA_MODELS"].split(",") if n.strip()]; unknown = [n for n in wanted if n not in MLX_CHECKPOINTS]; assert not unknown, "unknown checkpoint alias: " + ", ".join(unknown); print("\n".join(snapshot_download(MLX_CHECKPOINTS[n]) for n in wanted))'

http:
	LAYA_HOST=$(HOST) LAYA_PORT=$(PORT) LAYA_MODELS=$(MODELS) HF_HUB_OFFLINE=1 $(BIN)/laya-mlx-http

mcp:
	LAYA_MODELS=$(MODELS) HF_HUB_OFFLINE=1 $(BIN)/laya-mlx-mcp

test:
	HF_HUB_OFFLINE=1 $(BIN)/python -m pytest -q tests server/tests

lint:
	$(BIN)/ruff check laya_mlx server/laya_mlx_server server/tests

# Publishable artifact for the serve layer. Upload is a separate, deliberate step.
build:
	cd server && $(if $(UV_INDEX),UV_DEFAULT_INDEX=$(UV_INDEX) )uv build --out-dir dist

clean:
	rm -rf $(VENV) .pytest_cache laya_mlx.egg-info dist server/dist server/*.egg-info
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
