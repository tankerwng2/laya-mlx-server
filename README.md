<a href="README.zh-CN.md">简体中文</a> | English

# laya-mlx-server

A service layer on top of [laya-mlx](https://github.com/mizorewww/laya-mlx): **HTTP** speaking the TypeSafe Jev wire protocol, plus **MCP over stdio**, shipped as its own distribution, `laya-mlx-server`. One forward pass per batch of questions, **0 output tokens**, fully local — no PyTorch runtime, no cloud API.

The model, weight conversion, benchmarks and the Snake demo belong to upstream and are deliberately not copied here:

- [Upstream README](https://github.com/mizorewww/laya-mlx#readme) — the model, Python API, quick start
- [BENCHMARKS.md](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md) — upstream's full method and every timing sample, measured on an **M3 Max**
- [Snake demo](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md) · [Snake speed](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_BENCHMARKS.md) · [Snake optimization](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_OPTIMIZATION.md)

## Where the boundary is

| This repo | Upstream |
|---|---|
| `server/laya_mlx_server/`: HTTP surface, MCP surface, shared runtime | `laya_mlx/`: model, tokenization, calibration, presets, router, snake |
| `laya-mlx-http` / `laya-mlx-mcp` entry points | `laya-mlx` / `laya-snake` entry points |
| Serve-layer contract tests (9) | Model and kernel tests |

There is **no** serve code inside `laya_mlx/`, so syncing upstream cannot conflict with this package. Currently synced to upstream **0.2.0** (adds `shortlist.py`); the pinned upstream Laya revision is `573e5b6`.

## Quick start

Requires **Apple Silicon + macOS 14+** (`mlx` publishes darwin/arm64 wheels only, so this server **cannot** run in a Linux container; on Linux use upstream `laya-serve`), plus [uv](https://docs.astral.sh/uv/) and roughly 1.5 GB of disk.

```bash
git clone git@github.com:tankerwng2/laya-mlx-server.git && cd laya-mlx-server
make install          # one venv with laya-mlx and laya-mlx-server
make prefetch       # one-time weights (678 MB); offline afterwards
make http           # http://127.0.0.1:8080
```

Once published, an environment that already has `laya-mlx` can just `pip install 'laya-mlx-server[mcp]'` without cloning this repository.

## Calling the API

```bash
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $LAYA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"state":{"message":"I was charged twice, please refund the duplicate."},
       "questions":{"refund":{"type":"noul","instructions":"Does the customer want a refund?"},
                    "dept":{"type":"choice","instructions":"Who should handle this?","criteria":["billing","technical","sales"]}}}'
```

> Request shape only. Every figure in this README was measured with **Chinese** input: this English text scores near zero on the default multilingual checkpoint, so send it to `english` instead (see Picking a checkpoint).

`type` takes three values: `noul` (yes/no), `choice` (pick one, labels in `criteria`), `score` (ordinal rubric; `criteria` is an ordered list of level **descriptions**, not min/max).

`GET /health/live` only says the process is up. `GET /health/ready` is the one that says a checkpoint is resident, and returns 503 otherwise.

## MCP

Stdio transport: no port, no token.

```json
{ "laya-mlx-server": { "type": "stdio", "command": "laya-mlx-mcp",
  "env": { "HF_HUB_OFFLINE": "1", "LAYA_MODELS": "multilingual" } } }
```

Three tools: `laya_predict`, `laya_preset` (`triage` / `email` / `guard` / `moderation` / `router`), `laya_status`.

## Configuration

| env var | meaning | default |
|---|---|---|
| `LAYA_HOST` / `LAYA_PORT` | HTTP bind | `127.0.0.1` / `8080` |
| `LAYA_API_KEY` | when set, requires `Authorization: Bearer`; unset means unauthenticated and logs a warning at startup | none |
| `LAYA_MODELS` | checkpoints to preload, comma separated; empty loads on first request | `multilingual` |
| `LAYA_STATE_MODE` | `flatten` renders object state as `key: value` lines; `json` keeps upstream byte-identical | `flatten` |
| `LAYA_DEVICE` / `LAYA_DTYPE` | `gpu`/`metal`/`cpu`; `float16`/`bfloat16`/`float32` | auto / `float16` |
| `LAYA_BATCH_SIZE` / `LAYA_MAX_LOADED` | questions per forward pass / resident checkpoints | `16` / `2` |
| `HF_HOME` / `HF_ENDPOINT` | checkpoint cache root / mirror endpoint | `~/.cache/huggingface` / `hf-mirror.com` (prefetch) |

## Measured on this machine

Environment: **Apple M5 Max** (`applegpu_g17s`, 128 GiB unified memory), macOS 27.0, Python 3.13.15, MLX 0.32.2, laya-mlx **0.2.0**. Method is upstream's own `benchmarks.worker` — isolated processes, timing on synchronised GPU completion, downloads and model loading excluded:

| bucket (mlx / float16, multilingual) | P50 / P95 | throughput |
|---|---|---|
| short q=10 | **9.04 / 9.37 ms** | **1108.3 q/s** |
| short q=50 | **37.90 / 41.23 ms** | **1307.0 q/s** |
| long q=1 | **12.05 / 12.70 ms** | 82.9 q/s |
| long q=10 | **160.21 / 469.63 ms** | 51.3 q/s |

> [BENCHMARKS.md](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md) states an environment of **M3 Max / macOS 27.2 / Python 3.12.13**, which is not this machine (M5 Max, macOS 27.0, Python 3.13.15), so its figures are not interchangeable with the table above: on the same harness and method this machine measures **9.04 ms** for multilingual f16 short q=10 against 27.39 ms there.

Serve-layer measurements, all with Chinese input on multilingual: `noul` **0.9953**, `dept` → `billing`; unauthenticated request **401**. MCP `triage`: `intent=refund`, `refund_requested` **0.9676**.

## Picking a checkpoint

Multilingual scores **English** short input very low: the same noul measures about **0.01** for `I was charged twice` against about **1.00** for `发票被重复扣款，请退款`. Use the `english` checkpoint for English traffic (`LAYA_MODELS=english`).

## flatten or json

Jev clients send `state` as an object, and `serialize_state` turns it into literal JSON before tokenizing; `flatten` keeps the field names and drops the punctuation. On **0.2.0** multilingual the two modes no longer flip the verdict: a three-field Chinese state measures `noul` **0.9997** flat vs **0.9966** json, both routing to `billing`. The upstream 0.1.0-era English flip (0.16 vs 0.69) does not reproduce. Set `LAYA_STATE_MODE=json` for byte-identical upstream parity.

## Installing from mainland China networks

| symptom | cause | fix |
|---|---|---|
| `uv`/`pip` dependency install: `Connection refused` | uv/reqwest adopt the macOS **system proxy** (a Clash-class client), which is refusing CONNECT to PyPI; curl does not read the system proxy | `make install UV_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`, or `NO_PROXY='*' make install` |
| `git clone` / `git push` times out | `github.com:443` is unreachable while `api.github.com`, `github.com:22` and `ssh.github.com:443` all connect | use SSH (this repo's remote already is) |
| weights will not download | `huggingface.co` is blocked | `make prefetch` uses `hf-mirror.com` by default; override with `HF_ENDPOINT` |

## Packaging and publish status

The name `laya-mlx-server` is **still free** on PyPI. `make build` is verified: a 20 KB wheel whose metadata reads `Requires-Dist: laya-mlx>=0.1` (the workspace source does not leak into metadata) and which bundles `LICENSE` and `NOTICE`. **Not published** — uploading is a separate step that needs your explicit go-ahead:

```bash
make build
uv publish --publish-dir server/dist     # token generated yourself on pypi.org
```

## Quality gates

`make test` → **150 passed, 1 skipped**, including the nine serve-layer contract tests, which need no weights and no network (they pass against an empty `HF_HOME` with `HF_HUB_OFFLINE=1`). `make lint` and `ruff format --check .` are clean. CI installs both packages, runs `pytest tests server/tests` and builds both wheels.

## Attribution and license

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The model and pretrained weights are by [Convai Innovations and Laya contributors](https://github.com/NandhaKishorM/laya); prompt construction, output formatting, language routing, email utilities and presets are adapted from upstream `laya` at pin `573e5b6`; the inference kernel is reimplemented in Apple MLX by [laya-mlx](https://github.com/mizorewww/laya-mlx). Model output probabilities are not a guarantee of correctness.
