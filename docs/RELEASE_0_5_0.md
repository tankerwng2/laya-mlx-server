# 0.5.0 — upstream v0.4.12 migration

This release aligns applicable host APIs with NandhaKishorM/laya
`ffc142012af63d015fa19f05f29f9c5ff329a2cc`: batch and long-text inference,
structured decisions, hooks, calibration, evaluation, Router, HTTP/MCP,
and framework/tool integrations. Native execution and explicit unsupported
backend boundaries are preserved.

Compatibility changes include upstream Router defaults, validation and output
semantics. The LlamaIndex adapter supports the actual current BaseSelector and
SingleSelection contracts. See MIGRATION_0_4.md for behavior and dependency details.

The 341-entry API audit and three-checkpoint FP32/FP16 numerical/stability campaign
passed. The qualification report records unchanged applicable upstream assertions,
real SDK HTTP checks, real optional-framework checks and clean-wheel inference.
Package 0.5.0 changes version metadata relative to the recorded 0.4.0 development
qualification builds; release gates verify the final distribution separately.

Performance comparisons use upstream PyTorch 2.14, matched FP32 and fixed inputs.
MLX wins all 18 P50 comparisons in the final matrix. Core ML's alternating-order
confirmation wins all 18 P50/P95/mean-throughput comparisons; the initial four
sequential regressions are retained in the report. Core ML adaptive graph and
shape caches trade more resident memory for lower warm-call latency. Cold calls,
other hardware and arbitrary workloads are outside these finite guarantees.

See QUALIFICATION_0_4_12.md and UPSTREAM_0412_PROGRESS.json for raw evidence and
limitations. Original historical benchmark and package receipts remain unchanged.

MLX preserves native precision, compilation, row batch limits and bounded
question-prefix caching. Local-only model resolution is supported by Agent and
Router; ordinary inference needs no PyTorch or Transformers.

Publication verified on 2026-10-20: GitHub Release and PyPI wheel/sdist hashes match, public wheel source matches the tag, and a public-index reinstall passes all three offline real-model checks. See `benchmarks/results/release-v050.json`.

Post-release CI runs exposed intermittent failures in the unchanged upstream idle-window timing test. Running host contracts before native model tests in a separate process passed one complete CI, but a subsequent run reproduced the timing failure; isolation has not eliminated it. No runtime, assertion or timeout was changed. Failure diagnostics also run the original upstream idle-window tests for comparison. [CI evidence](https://github.com/mizorewww/laya-mlx/actions/runs/39354961828).

The original pinned upstream suite reproduced the same idle-window failure on the same CI runner; all 254 native/integration tests passed. This timing instability remains documented in [CI evidence](https://github.com/mizorewww/laya-mlx/blob/main/docs/CI_TIMING.md).
