[English](README.md) | 简体中文

# laya-mlx-server

在 [laya-mlx](https://github.com/mizorewww/laya-mlx) 之上加一层对外服务：**HTTP**（兼容 TypeSafe Jev 线协议）与 **MCP stdio**，打包成独立发行版 `laya-mlx-server`。一次前向传播给出结构化判定，**0 个输出 token**，全程本地推理，无 PyTorch、无云端 API。

模型内核、权重转换、benchmark、Snake demo 仍归上游，本仓库不再复制那些内容：

- [上游 README](https://github.com/mizorewww/laya-mlx#readme) —— 模型、Python API、快速开始
- [BENCHMARKS.md](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md) —— 上游在 **M3 Max** 上的完整方法与全部计时样本
- [Snake 说明](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md) · [Snake 速度](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_BENCHMARKS.md) · [Snake 优化](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_OPTIMIZATION.md)

## 边界

| 归本仓库 | 归上游 |
|---|---|
| `server/laya_mlx_server/`：HTTP 面、MCP 面、共享运行时 | `laya_mlx/`：模型、tokenization、校准、presets、router、snake |
| `laya-mlx-http` / `laya-mlx-mcp` 两个入口 | `laya-mlx` / `laya-snake` 两个入口 |
| serve 层的契约测试（9 个） | 模型与内核测试 |

`laya_mlx/` 里**没有**任何服务代码，所以同步上游不会与这套服务冲突。当前已同步到上游 **0.2.0**（新增 `shortlist.py`），上游 Laya 引用修订 pin 在 `573e5b6`。

## 快速开始

前提：**Apple Silicon + macOS 14+**（`mlx` 只发布 darwin/arm64 wheel，所以这套服务**不能**放进 Linux 容器；Linux 请用上游 `laya-serve`），外加 [uv](https://docs.astral.sh/uv/) 与约 1.5 GB 磁盘。

```bash
git clone git@github.com:tankerwng2/laya-mlx-server.git && cd laya-mlx-server
make install          # 一个 venv，同时装 laya-mlx 与 laya-mlx-server
make prefetch        # 拉一次权重（678 MB），之后可全程离线
make http            # http://127.0.0.1:8080
```

包发布后，已有 `laya-mlx` 的环境可以直接 `pip install 'laya-mlx-server[mcp]'`，不必 clone 仓库。

## 调用 API

```bash
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $LAYA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"state":{"message":"发票被重复扣款，请退款。"},
       "questions":{"refund":{"type":"noul","instructions":"客户是否要求退款？"},
                    "dept":{"type":"choice","instructions":"由哪个部门处理？","criteria":["billing","technical","sales"]}}}'
```

`type` 只有三种：`noul`（是不是）、`choice`（多选一，选项写在 `criteria`）、`score`（按档位打分，`criteria` 是从低到高的**文字**档位，不是 min/max）。

`GET /health/live` 只看进程活着；`GET /health/ready` 才说明检查点已常驻（否则 503）。

## MCP

走 stdio，不需要端口也不需要 token：

```json
{ "laya-mlx-server": { "type": "stdio", "command": "laya-mlx-mcp",
  "env": { "HF_HUB_OFFLINE": "1", "LAYA_MODELS": "multilingual" } } }
```

工具三个：`laya_predict`、`laya_preset`（`triage` / `email` / `guard` / `moderation` / `router`）、`laya_status`。

## 配置

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `LAYA_HOST` / `LAYA_PORT` | HTTP 绑定地址与端口 | `127.0.0.1` / `8080` |
| `LAYA_API_KEY` | 设定后要求 `Authorization: Bearer`；未设定则不鉴权并在启动时告警 | 无 |
| `LAYA_MODELS` | 预加载的检查点，逗号分隔；为空则首次请求时加载 | `multilingual` |
| `LAYA_STATE_MODE` | `flatten` 把对象 state 渲染成 `key: value` 行；`json` 与上游逐字节一致 | `flatten` |
| `LAYA_DEVICE` / `LAYA_DTYPE` | `gpu`/`metal`/`cpu`；`float16`/`bfloat16`/`float32` | auto / `float16` |
| `LAYA_BATCH_SIZE` / `LAYA_MAX_LOADED` | 单次前向题数 / 常驻检查点数 | `16` / `2` |
| `HF_HOME` / `HF_ENDPOINT` | 权重缓存根 / 镜像端点 | `~/.cache/huggingface` / `hf-mirror.com`（prefetch） |

## 本机实测

环境：**Apple M5 Max**（`applegpu_g17s`，128 GiB 统一内存）、macOS 27.0、Python 3.13.15、MLX 0.32.2、laya-mlx **0.2.0**。口径用上游的 `benchmarks.worker`（独立进程、GPU 同步完成计时、排除下载与加载）：

| 桶（mlx / float16，multilingual） | P50 / P95 | 吞吐 |
|---|---|---|
| short q=10 | **9.04 / 9.37 ms** | **1108.3 q/s** |
| short q=50 | **37.90 / 41.23 ms** | **1307.0 q/s** |
| long q=1 | **12.05 / 12.70 ms** | 82.9 q/s |
| long q=10 | **160.21 / 469.63 ms** | 51.3 q/s |

> [BENCHMARKS.md](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md) 标注的测量环境是 **M3 Max / macOS 27.2 / Python 3.12.13**，与本机（M5 Max、macOS 27.0、Python 3.13.15）不符，不能与上表互换；同一 harness 同口径下本机 multilingual f16 short q=10 为 **9.04 ms**，该文件为 27.39 ms。

服务面实测：中文 `noul` **0.9953**、`dept` → `billing`；无 token 请求 **401**。MCP `triage` 实测 `intent=refund`、`refund_requested` **0.9676**。

## 选检查点

multilingual 对**英文**短输入给分极低：同一道 `noul`，`I was charged twice` 实测约 **0.01**，`发票被重复扣款，请退款` 约 **1.00**。英文流量请用 `english` 检查点（`LAYA_MODELS=english`）。

## flatten 还是 json

Jev 客户端的 `state` 通常是对象，`serialize_state` 会把它变成字面 JSON 再 tokenization；`flatten` 保留字段名、去掉 JSON 标点。**0.2.0** 的 multilingual 上两种模式已不再翻转判定：三字段中文 state 实测 `noul` flat **0.9997** / json **0.9966**，`dept` 同为 `billing`。上游 0.1.0 时代英文短输入的翻转（0.16 / 0.69）在 0.2.0 不复现。需要与上游逐字节一致时设 `LAYA_STATE_MODE=json`。

## 大陆网络安装（踩过的坑）

| 现象 | 原因 | 解法 |
|---|---|---|
| `uv`/`pip` 装依赖 `Connection refused` | macOS **系统代理**（Clash 类客户端）会被 uv/reqwest 自动采用，而它对 PyPI 的 CONNECT 正在拒；curl 不读系统代理 | `make install UV_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`，或临时 `NO_PROXY='*' make install` |
| `git clone` / `git push` 超时 | `github.com:443` 不通，但 `api.github.com`、`github.com:22`、`ssh.github.com:443` 全通 | 改用 SSH（本仓库 remote 已是 SSH） |
| 权重下不动 | `huggingface.co` 被墙 | `make prefetch` 默认走 `hf-mirror.com`，可用 `HF_ENDPOINT` 覆盖 |

## 打包与发布状态

`laya-mlx-server` 这个名字在 PyPI **尚空闲**。`make build` 已验证可产出 20 KB 的 wheel，元数据为 `Requires-Dist: laya-mlx>=0.1`（workspace 的本地来源不会泄漏进元数据），并打包 `LICENSE` 与 `NOTICE`。**尚未发布**——上传是需要你明确点头的独立动作：

```bash
make build
uv publish --publish-dir server/dist     # token 在 pypi.org 自行生成
```

## 质量门禁

`make test` → **150 passed, 1 skipped**（含 serve 层 9 个契约测试，后者零权重、零网络：空 `HF_HOME` + 强制离线仍全过）。`make lint` 与 `ruff format --check .` 干净。设备覆盖要分开看：CI 在 env 里钉了 `LAYA_MLX_TEST_DEVICE=cpu`，所以 **CI 只跑 CPU**；Metal（默认设备）由本机 `make test` 覆盖，两个设备各连跑十轮均通过。分块前向在 Metal 上因归约顺序不同会有 1e-4 漂移，所以相关断言按判定严格相等、浮点容差 `2e-4` 写。

## 许可与致谢

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。模型与预训练权重来自 [Convai Innovations 与 Laya 贡献者](https://github.com/NandhaKishorM/laya)；提示构造、输出格式、语言路由、邮件工具与 presets 改编自上游 `laya`（pin `573e5b6`）；推理内核为 [laya-mlx](https://github.com/mizorewww/laya-mlx) 用 Apple MLX 重新实现。模型输出的概率不等于答案必然正确。
