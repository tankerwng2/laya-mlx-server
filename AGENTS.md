# laya-mlx（fork）作业须知

- 上游＝`mizorewww/laya-mlx`（remote `upstream`，push 已禁）；本仓库＝镜像层＋serve 层。
- **main 有 ruleset `main-protection`**：直接 push 会被拒，必须走 PR，且三项检查全绿＝`test (cpu)`／`test (metal)`／`sync freshness (mirror layer == upstream/main)`。
- 镜像层判据（ci.yml 的 pathspec）：`laya_mlx`（除 `email.py`/`router.py`）、`benchmarks`、`BENCHMARKS.md`、`NOTICE`、`LICENSE`、`examples`、`experiments`、`scripts`、`docs`、`tests`（除 `test_router.py`/`test_runtime.py`）必须与 `upstream/main` **逐字节一致**；fork-owned＝`Makefile`、`README*.md`、`pyproject.toml`、`uv.lock`、`.github/*`、`server/*`、`laya_mlx/email.py`、`laya_mlx/router.py`、`tests/test_router.py`、`tests/test_runtime.py`。两份清单必须同步改（`ci.yml` 与 `upstream-sync.yml`）。
- 追平上游＝跑 `upstream-sync` workflow（每 6 小时排程＋可手动 dispatch），它按上面的路径策略开 sync PR。**该作业必须有 `permissions.workflows: write`**——上游改 `.github/workflows/*` 时，缺这个 scope 会让 push 被拒（10-17 起连挂三次，镜像层落后 6 个提交，`sync freshness` 常红）。
- 手动追平的等价做法：从 main 开 `sync/upstream-<sha>`，`git merge upstream/main`，冲突按路径策略解（fork-owned 取 ours，其余取 theirs），再逐文件 `git checkout main -- <fork-owned>`，然后断言镜像层零差异、`uv lock` 复算，开 PR。
