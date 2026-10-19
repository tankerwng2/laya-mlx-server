# laya-mlx（fork）作业须知

<!-- ctx:inherits global=~/.omp/agent/AGENTS.md+RULES.md ws=~/Workspace/AGENTS.md -->
<!-- ctx:registered 10-20 全局索引行=~/.omp/agent/AGENTS.md#项目索引 -->
<!-- ctx:detail vault=项目/laya-mlx learn=本项目bank -->

预算自述（闸＝`gate-check` 闸2）：laya-mlx/AGENTS.md ≤8 KB / ≤40 行、单条 ≤300 字符。

- 上游＝`mizorewww/laya-mlx`（remote `upstream`，push 已禁）；本仓库＝镜像层＋serve（人话：把模型跑起来对外提供接口） 层。
- **main 有 ruleset `main-protection`**：直接 push 会被拒，必须走 PR，且三项检查全绿＝`test (cpu)`／`test (metal)`／`sync freshness (mirror layer == upstream/main)`。
- 镜像层判据（`ci.yml` 的 pathspec）：`laya_mlx`（全部）、`benchmarks`、`BENCHMARKS.md`、`NOTICE`、`LICENSE`、`examples`、`experiments`、`scripts`、`docs`、`tests`（除 `test_router.py`/`test_runtime.py`）必须与 `upstream/main` **逐字节一致**；fork-owned＝`Makefile`、`README*.md`、`pyproject.toml`、`uv.lock`、`.github/*`、`server/*`、`tests/test_router.py`、`tests/test_runtime.py`。两份清单必须同步改（`ci.yml` 与 `upstream-sync.yml`）。
- `laya_mlx/email.py`、`laya_mlx/router.py` 于 10-18 交还镜像层：fork 自分叉点 `0a85951` 起只改了 email_questions 的导入位置与 `_repo_str` 包装，上游已含等价实现；继续标 fork-owned 会让上游新增的镜像层测试（`test_decision_helpers`／`test_router_lifecycle`／`test_upstream_sync`／`test_email`）在 metal 腿炸 41 项。加新文件进 fork-owned 之前先跑一次全量 pytest。
- 本地 M5 Max 的 Metal fp32 漂移 ≈1.3e-4，比 `tests/test_parallel_layout.py` 的 atol 2e-5（CPU 口径）大 ⇒ 本地单跑该测试会红，CI（人话：提交后自动跑的检查流水线）的 macos-26 metal 腿不复现；判绿以 CI 两腿为准，别为此放宽容差。
- 追平上游＝跑 `upstream-sync` workflow（每 6 小时排程＋可手动 dispatch），它按上面的路径策略开 sync PR。**该作业必须有 `permissions.workflows: write`**——上游改 `.github/workflows/*` 时，缺这个 scope 会让 push 被拒（10-17 起连挂三次，镜像层落后 6 个提交，`sync freshness` 常红）。
- 手动追平的等价做法：从 main 开 `sync/upstream-<sha>`，`git merge upstream/main`，冲突按路径策略解（fork-owned 取 ours，其余取 theirs），再逐文件 `git checkout main -- <fork-owned>`，然后断言镜像层零差异、`uv lock` 复算，开 PR。
