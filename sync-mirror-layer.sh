#!/usr/bin/env bash
# sync-mirror-layer.sh — 满足 laya-mlx-server 的 `sync freshness` 仓规（镜像层与上游逐字节一致）
# 用途: 本 fork＝「镜像层（上游模型代码）＋ 自研外壳」双层结构；ci.yml 的 sync-freshness 断言
#       镜像路径必须与 mizorewww/laya-mlx main 逐字节一致，自研文件（fork-owned）不受限。
# 输入: 无参数＝在当前分支执行；`--dry`＝只打印将要覆盖/恢复的路径
# 输出: 每行一条动作结论；末行打印镜像层差异（应为 0）
# 日志: 无（git 历史即记录；合并提交信息写明策略）
# 用法: bash sync-mirror-layer.sh [--dry]
# 退出码: 0=镜像层已对齐; 1=仍有差异（需人工看）
set -uo pipefail
DRY=0; test "${1:-}" = "--dry" && DRY=1
BR="$(git rev-parse --abbrev-ref HEAD)"
git fetch -q upstream main

# 与 .github/workflows/upstream-sync.yml 的 is_mirrored() 保持同一口径（单一真源＝那份工作流）
is_mirrored() {
	case "$1" in
		laya_mlx/* | benchmarks/* | BENCHMARKS.md | NOTICE | LICENSE \
		| examples/* | experiments/* | scripts/* | docs/*) return 0 ;;
		tests/*)
			case "$1" in
				tests/test_router.py | tests/test_runtime.py) return 1 ;;
				*) return 0 ;;
			esac ;;
		*) return 1 ;;
	esac
}

if [ "${DRY}" -eq 1 ]; then
	git diff --name-only main upstream/main | while IFS= read -r f; do
		is_mirrored "$f" && echo "取上游（镜像层）: $f" || echo "保我方（自研）: $f"
	done
	exit 0
fi

git merge -q --no-edit upstream/main 2>/dev/null || git merge -q --no-edit --allow-unrelated-histories upstream/main
# 合并结果按路径策略定调：镜像层＝上游内容；自研文件＝我方内容
git checkout -q upstream/main -- laya_mlx benchmarks BENCHMARKS.md NOTICE LICENSE examples experiments scripts docs tests 2>/dev/null
while IFS= read -r f; do
	is_mirrored "$f" && continue
	git cat-file -e "main:${f}" 2>/dev/null && git checkout -q main -- "${f}" && echo "保我方: ${f}"
done < <(git diff --name-only main upstream/main)
if git diff --cached --quiet && git diff --quiet; then
	echo "无新增改动（镜像层本已对齐）"
else
	git add -A
	git commit -q -m "sync: 对齐镜像层到 upstream/main（按路径策略：镜像取上游、自研保我方）

口径＝.github/workflows/upstream-sync.yml 的 is_mirrored()；
自研文件（README/pyproject/server/.omp/根 ARCHITECTURE.md 等）全部保留我方版本。"
	echo "已生成对齐提交 @ $(git rev-parse --short=10 HEAD)"
fi
D="$(git diff --name-only upstream/main -- laya_mlx benchmarks BENCHMARKS.md NOTICE LICENSE examples experiments scripts docs tests ':!laya_mlx/email.py' ':!laya_mlx/router.py' ':!tests/test_router.py' ':!tests/test_runtime.py' | wc -l | tr -d ' ')"
echo "镜像层与上游差异文件数＝${D}（0＝sync freshness 应通过）｜当前分支=${BR}"
test "${D}" = "0"
