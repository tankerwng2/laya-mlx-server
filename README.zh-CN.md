# Laya-MLX

![Laya MLX 实际运行记录，原速回放](docs/assets/snake-demo.gif)

**在 Apple Silicon 上本地运行开放权重的结构化决策模型。**

单个短问题端到端中位耗时 **13.4 ms**；multilingual 检查点为 **7.4 ms**。**0 个输出 token**，原生 MLX，无 PyTorch / Transformers 推理依赖，无云端 API。

[English](README.md) · [完整 benchmark](BENCHMARKS.md) · [Snake 使用说明](docs/SNAKE_DEMO.md) · [30 秒 MP4](docs/assets/snake-demo.mp4)

GIF 使用真实游戏记录按原始时间戳渲染。每步都调用 Laya，界面显示循环路径安全层及其接管次数。上面的 13.4 / 7.4 ms 来自**单问题 API 基准**，并非每步批量回答三个问题的 Snake 帧耗时；游戏速度见[独立报告](docs/SNAKE_BENCHMARKS.md)。

## 快速开始

```bash
pip install laya-mlx
```

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-multilingual-mlx")
result = agent.predict(
    "发票被重复扣款，请退款。",
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical", "sales"],
        }
    },
)
print(result["answers"]["department"])
```

运行贪吃蛇：

```bash
pip install 'laya-mlx[demo]'
hf download aac6fef/laya-multilingual-mlx
laya-snake
```

提前下载一次权重，游戏运行期间完全本地推理。终端至少 104 列 × 35 行；空格暂停、↑/↓ 调速、R 重开、Q 退出。`--max-speed` 持续满速运行，每一步等待新的模型结果。

`laya-snake --optimize --max-speed` 启用已验证的编译与前缀复用路径。同轮成对测试中，2,400 步达到 **75.40 步/秒**，零死亡、安全接管 2 次，比 eager 基线快约 **6.5%**。[完整游戏表现、优化测量和一致性证据](docs/SNAKE_OPTIMIZATION.md)。

## M3 Max 实测

| FP16，端到端 | Laya 421M | Multilingual 322M |
|---|---:|---:|
| 单个短问题 P50 | **13.42 ms** | **7.39 ms** |
| 单个短问题 P95 | **13.92 ms** | **7.79 ms** |
| 50 问题吞吐量 | **146.8 q/s** | **395.0 q/s** |
| 单个短问题 MLX 峰值分配 | **943.6 MiB** | **687.6 MiB** |

硬件为 M3 Max（40 核 GPU、128 GiB 内存）。计时包含提示准备、tokenization、张量构建、GPU 同步推理、校准及结果格式化，排除模型加载。50 问题吞吐量使用 `batch_size=64`，公开 API 默认为 16。

**移植一致性：**三个检查点在 FP32 和 FP16 下均通过 **63/63** 验证问题的上游 argmax 对齐，合计 378/378；每个配置各执行 100 次重复调用，结果有限、确定，测得活跃内存增长为零。它验证移植保真度，不代表所有实际问题都能答对。[完整误差和原始记录](BENCHMARKS.md)。

`choice` 返回分类概率，`score` 返回有序评分，`noul` 返回 P(true)。每个问题作为独立行经过双向编码器；不宣称任意问题可以复用同一份 state hidden states。本项目是独立 MLX 移植，并非 Convai Innovations 官方发布。

## 支持的检查点

| 检查点 | 编码器 | 参数量 | 最大上下文 |
|---|---|---:|---:|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 |
| `convaiinnovations/laya-multilingual` | mmBERT-base | 322M | 1,024 |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 421M | 1,024 |

上下文预算包含问题、选项和输入状态。中文等非英语输入应使用 multilingual 检查点。本项目实现推理与权重转换；RLCD 训练和微调继续使用上游项目。

已转换的 FP16 MLX 权重发布在 Hugging Face，可直接传给 `laya.load(...)`：

- [aac6fef/laya-mlx](https://huggingface.co/aac6fef/laya-mlx)
- [aac6fef/laya-multilingual-mlx](https://huggingface.co/aac6fef/laya-multilingual-mlx)
- [aac6fef/laya-typed-decisions-mlx](https://huggingface.co/aac6fef/laya-typed-decisions-mlx)

例如：`laya.load("aac6fef/laya-multilingual-mlx")`。每个模型仓库都包含模型卡、测试结果、来源、许可证和文件校验清单。三个仓库共 36 个文件均已通过严格远端校验；固定版本与权重哈希见 [hub-publication.json](benchmarks/results/hub-publication.json)。

## 安装与运行

需要 Apple silicon Mac、macOS 14+ 和 Python 3.11+。本机实测环境为 M3 Max（40 核 GPU、128 GB 内存）、macOS 27.2、Python 3.12.13、MLX 0.32.2。MLX 0.32.2 提供 macOS 14 / 15 / 26 的 wheel，本机选择了 26 构建；未在这台机器上实测旧系统。

```bash
gh repo clone mizorewww/laya-mlx
cd laya-mlx
uv sync
uv run python examples/quickstart.py
```

或从 GitHub 直接安装：

```bash
python -m pip install 'git+https://github.com/mizorewww/laya-mlx.git'
```

Python 示例：

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-multilingual-mlx")
result = agent.predict(
    "发票被重复扣款，请今天退款。",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": ["billing", "technical", "sales"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
)
print(result["answers"])
```

默认使用 FP16。需要更接近原版 FP32 的数值时使用 `dtype="float32"`。`batch_size=16` 控制每次计算的问题数，更多问题会分批处理。概率按原版格式保留四位小数；不同精度可能造成小幅差异，实测误差见 benchmark 报告。

命令行支持文本或 JSON 状态：

```bash
uv run laya-mlx predict \
  --model aac6fef/laya-multilingual-mlx \
  --state '发票被重复扣款，请退款。' \
  --questions examples/questions.json
```

本仓库已下载的权重位于 `models/` 时，将 `--model` 改成相应本地目录即可避免再次下载。

## 转换权重

```bash
uv run laya-mlx convert \
  --model convaiinnovations/laya \
  --dtype float16 \
  --output models/laya-mlx-fp16
```

转换后可以通过 `laya.load("./models/laya-mlx-fp16")` 直接加载。输出目录包含模型、配置、tokenizer 和来源元数据。已有目录不会被覆盖，模型权重不会提交到 GitHub。

原始检查点本身存储的是 FP16 权重；这里的转换调整参数命名与计算精度，不涉及重新训练或低比特量化。

## 路由、测试和 benchmark

`Router`、`triage_questions`、`email_questions`、`guard_questions`、`moderation_questions` 等接口保留上游用法，将导入名改为 `laya_mlx` 即可。typed-decisions 检查点可通过 `task="typed_decisions"` 显式指定；`Router(preload=True)` 可预加载三个模型。

详细的 API、测试和复现命令见 [英文 README](README.md)。[BENCHMARKS.md](BENCHMARKS.md) 包含本机 PyTorch MPS FP32、MLX FP32 与 MLX FP16 的端到端 P50/P95、吞吐量、内存、数值一致性、重复运行和固定抽样分类测试。所有原始测量数据位于 [benchmarks/results](benchmarks/results)，GPU 测试应串行运行。

[初步性能研究](docs/PERFORMANCE_RESEARCH.md) 分析实现、基准和 MLX 源码。针对“能否再快一个数量级”，另有两份深入报告：

- [数学分析](docs/MATH_10X_RESEARCH.md)：计算预算、带宽条件下界、真实权重谱、精确复用，以及蒸馏学生模型的设计空间。
- [工程实测](docs/ENGINEERING_10X_RESEARCH.md)：编译、量化、最后一层输出裁剪、自定义 Metal 核与矩阵乘法实验。

[experiments/](experiments) 保存研究脚本和原始数据。发布版本的结果见 [BENCHMARKS.md](BENCHMARKS.md)，各实验变体的耗时与数值一致性单独记录。

目前证据不支持相同检查点下普遍再快 10 倍。部分场景的逐轮配对中位加速约为 1.03–1.08 倍；误差区间、量化保真结果和自定义 Metal 核的实测详见工程报告。

这是独立的 MLX 移植，模型能力及其限制来自上游；模型输出概率不等于答案必然正确。采用 Apache-2.0，原作者与移植说明见 [NOTICE](NOTICE)。

## 作为服务使用（HTTP / MCP）

上游只提供 Python API；这里加了两面对外接口，共用同一个运行时，仍然零 PyTorch。它们放在 `server/` 目录、打包成独立发行版 **`laya-mlx-server`**，不再往 `laya_mlx/` 里塞文件，所以同步上游新版时不会和这套服务代码冲突。`laya_mlx/` 已同步到上游 **0.2.0**（新增 `shortlist.py`），上游 Laya 引用修订 pin 在 `573e5b6`。

前提：**Apple Silicon + macOS**（`mlx` 只在 darwin/arm64 有 wheel，所以这套服务不能放进 Linux 容器；Linux 上请改用上游 `laya-serve`）。另外需要 [uv](https://docs.astral.sh/uv/) 和约 1.5 GB 磁盘。

```bash
git clone git@github.com:tankerwng2/laya-mlx-server.git && cd laya-mlx-server
make install          # 建 .venv，装 laya-mlx + laya-mlx-server（含 mcp）
make prefetch        # 拉一次权重（约 678 MB），之后可全程离线
make http            # http://127.0.0.1:8080
```

已经装好 `laya-mlx` 的环境可以直接 `pip install 'laya-mlx-server[mcp]'`（发布后），不必 clone 仓库。大陆网络如果 `files.pythonhosted.org` 连不上，加镜像：`make install UV_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`；权重默认走 `hf-mirror.com`，`HF_ENDPOINT` 可覆盖。

调用（`state` 可以是字符串或 JSON 对象）：

```bash
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $LAYA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"state":{"message":"客户被重复扣款两次，要求全额退款"},
       "questions":{"refund":{"type":"noul","instructions":"客户是否要求退款"},
                    "dept":{"type":"choice","instructions":"归属部门","criteria":["billing","technical","sales"]}}}'
```

`type` 只有三种：`noul`（是不是）、`choice`（多选一，选项写在 `criteria`）、`score`（按档位打分，`criteria` 是从低到高的**文字**档位）。

给 agent 用则走 MCP stdio，不需要端口和 token：

```json
{ "laya-mlx-server": { "type": "stdio", "command": "laya-mlx-mcp",
  "env": { "HF_HUB_OFFLINE": "1", "LAYA_MODELS": "multilingual" } } }
```

工具三个：`laya_predict`、`laya_preset`（`triage` / `email` / `guard` / `moderation` / `router`）、`laya_status`。

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `LAYA_HOST` / `LAYA_PORT` | HTTP 绑定 | `127.0.0.1` / `8080` |
| `LAYA_API_KEY` | 设置后要求 `Authorization: Bearer`；不设则无鉴权并告警 | 无 |
| `LAYA_MODELS` | 预加载检查点，逗号分隔 | `multilingual` |
| `LAYA_STATE_MODE` | `flatten` 把对象 state 渲染成 `key: value`；`json` 保持 upstream 原样 | `flatten` |
| `LAYA_DEVICE` / `LAYA_DTYPE` | `gpu`/`metal`/`cpu`；`float16`/`bfloat16`/`float32` | auto / `float16` |
| `LAYA_BATCH_SIZE` / `LAYA_MAX_LOADED` | 单次前向题数 / 常驻检查点数 | `16` / `2` |

**为什么默认 flatten**：Jev 客户端的 `state` 通常是对象，而 `serialize_state` 会把它变成字面 JSON 再 tokenization；展平成 `key: value` 行保留字段名、去掉 JSON 标点。**0.2.0** 的 multilingual 上两种模式已不再翻转判定：三字段中文 state 实测 `noul` flat **0.9997** / json **0.9966**，`dept` 同为 `billing`（0.9937 / 0.9987）。0.1.0 时代英文短输入的翻转（0.16 / 0.69）在 0.2.0 不复现。需要与 upstream 逐字节一致时设 `LAYA_STATE_MODE=json`。

**选检查点**：multilingual 对英文短输入给分极低——同一道 `noul`，`I was charged twice` 实测约 **0.01**，`发票被重复扣款，请退款` 约 **1.00**。英文场景用 `english` 检查点（`LAYA_MODELS=english`）。
