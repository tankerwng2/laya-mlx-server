# laya-mlx-server

Self-hosted **HTTP** and **MCP** servers for [laya-mlx](https://pypi.org/project/laya-mlx/): native MLX typed decisions on Apple silicon. One forward pass, no text generation, no cloud API.

The HTTP surface speaks the TypeSafe Jev wire format (`POST /v1/systemone`), so an existing Jev client can point at it by changing the base URL.

Requires **Apple Silicon + macOS 14+**. `mlx` publishes darwin/arm64 wheels only, so this server cannot run in a Linux container; on Linux use upstream `laya-serve`.

## Install and run

```bash
pip install laya-mlx-server          # pip install laya-mlx-server[mcp] for the MCP surface
laya-mlx-http                        # http://127.0.0.1:8080
```

Weights download once on first request (~678 MB for multilingual) and are cached under `~/.cache/huggingface`. Set `HF_ENDPOINT=https://hf-mirror.com` where huggingface.co is unreachable.

```bash
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $LAYA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"state":{"message":"I was charged twice, please refund the duplicate."},
       "questions":{"refund":{"type":"noul","instructions":"Does the customer want a refund?"},
                    "dept":{"type":"choice","instructions":"Who should handle this?","criteria":["billing","technical","sales"]}}}'
```

`type` is one of `noul` (yes/no), `choice` (pick one, labels in `criteria`), `score` (ordinal rubric; `criteria` is an ordered list of level descriptions, not min/max).

## MCP

Runs over stdio: no port, no token.

```json
{ "laya-mlx-server": { "type": "stdio", "command": "laya-mlx-mcp",
  "env": { "HF_HUB_OFFLINE": "1", "LAYA_MODELS": "multilingual" } } }
```

Tools: `laya_predict`, `laya_preset` (`triage` / `email` / `guard` / `moderation` / `router`), `laya_status`.

## Configuration

| env var | meaning | default |
|---|---|---|
| `LAYA_HOST` / `LAYA_PORT` | HTTP bind | `127.0.0.1` / `8080` |
| `LAYA_API_KEY` | when set, requires `Authorization: Bearer`; unset means unauthenticated and logs a warning | none |
| `LAYA_MODELS` | checkpoints to preload, comma separated; empty loads on first request | `multilingual` |
| `LAYA_STATE_MODE` | `flatten` renders object state as `key: value` lines; `json` keeps upstream byte-identical | `flatten` |
| `LAYA_DEVICE` / `LAYA_DTYPE` | `gpu`/`metal`/`cpu`; `float16`/`bfloat16`/`float32` | auto / `float16` |
| `LAYA_BATCH_SIZE` / `LAYA_MAX_LOADED` | questions per forward pass / resident checkpoints | `16` / `2` |
| `HF_HOME` | checkpoint cache root | `~/.cache/huggingface` |

**Why flatten is the default:** Jev clients send `state` as an object, and `serialize_state` turns it into literal JSON before tokenizing; flattening keeps the field names and drops the punctuation. On **0.2.0** multilingual the two modes no longer flip the verdict — a three-field Chinese state measures `noul` **0.9997** flat vs **0.9966** json, both routing to `billing`. The 0.1.0-era English flip (0.16 vs 0.69) does not reproduce. Set `LAYA_STATE_MODE=json` for byte-identical upstream parity.

**Picking a checkpoint:** multilingual scores English short input very low — the same noul measures about **0.01** for `I was charged twice` against about **1.00** for `发票被重复扣款，请退款`. Use the `english` checkpoint for English traffic (`LAYA_MODELS=english`).

## Attribution

Derived from [Laya](https://github.com/NandhaKishorM/laya) and [laya-mlx](https://github.com/mizorewww/laya-mlx), Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model output probabilities are not a guarantee of correctness.
