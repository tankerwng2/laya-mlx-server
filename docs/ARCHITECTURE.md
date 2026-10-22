# laya-mlx 架构图与流程图（10-20 补真图）

> 定位：在 Apple 芯片上用 MLX 原生跑 Laya 类型化决策模型（typed decision＝一次前向输出概率而非生成文字）的 Python 包＋MCP 服务。155 个跟踪文件＝`benchmarks/`(44)＋`laya_mlx/`(23)＋`server/`(10)。

## 一、架构图（组件与依赖）

```mermaid
flowchart LR
 subgraph client["调用方"]
  OMP["omp / Codex 等客户端<br/>走 MCP（人话：让程序调用模型的标准接口）"]
  QS["examples/quickstart.py<br/>脚本直调"]
 end
 subgraph pkg["laya_mlx/（核心包）"]
  LOAD["模型加载<br/>权重读取＋MLX 张量装配（人话：多维数组运算）"]
  FWD["一次前向推理<br/>数据进去、概率出来（人话：只算一次，不生成文字）"]
  TOK["输入编码<br/>状态文本＋问题拼装"]
  OUT["结果解析<br/>choice / score / noul（选项/打分/是否）"]
 end
 subgraph svc["服务端那一块 server/（人话：一直开着等请求的那块代码）"]
  SRV["laya-mlx-mcp<br/>常驻进程（人话：一直开着等请求的程序）"]
  ST["健康状态<br/>设备、数据类型 dtype、显存占用（人话：显卡自己的内存）"]
 end
 subgraph bench["benchmarks/（44 个文件）"]
  DL["download.py<br/>权重下载"]
  RUN["run.py / worker.py<br/>跑分调度"]
  ACC["accuracy.py / validate.py<br/>准确率与校验"]
  RPT["report.py / snake_*.py<br/>榜单与对比"]
 end
 MD["~/.cache 权重目录"]
 OMP --> SRV --> TOK --> FWD
 QS --> LOAD
 MD --> LOAD --> FWD
 FWD --> OUT --> SRV --> OMP
 ST --> SRV
 DL --> MD
 RUN --> FWD
 RUN --> ACC --> RPT
```

## 二、流程图（一次类型化判断请求）

```mermaid
sequenceDiagram
 participant C as 客户端(omp)
 participant S as laya-mlx-mcp
 participant M as MLX 推理核
 participant B as benchmarks
 使用者->>C: 问"这段日志是不是崩溃前兆？"
 C->>S: MCP 请求（状态文本＋问题类型 noul）
 S->>M: 编码输入，加载权重（首次＝冷启动，人话：从零起来最慢那回）
 M-->>S: 返回各标签概率（如 是=0.83）
 S-->>C: 结构化结果＋用量统计
 C->>C: 概率落 [0.3,0.7]＝判不准，升级大模型或问人
 Note over B: 版本发布前
 B->>M: 批量跑题集（run.py → worker.py）
 B->>B: accuracy.py 算分 → report.py 出榜
```

## 三、关键目录

| 路径 | 作用 |
|---|---|
| `laya_mlx/`(23) | 加载、编码、前向、结果解析 |
| `server/`(10) | MCP 服务进程与健康状态 |
| `benchmarks/`(44) | 下载、跑分、校验、榜单 |
| `experiments/`、`examples/` | 成本/能力实验与快速上手样例 |
| `tests/`(11) | 单元与集成测试 |

## 四、怎么跑 / 怎么验

```bash
uv tool install laya-mlx        # 安装（提供 laya-mlx-mcp 命令）
laya-mlx-mcp --help
bun test                        # 或 python -m pytest tests/
python -m benchmarks.run        # 跑分（需先 download.py 拉权重）
```
