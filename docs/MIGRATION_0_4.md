# 0.5.0 migration to upstream v0.4.12

The host API targets upstream `ffc142012af63d015fa19f05f29f9c5ff329a2cc`.
Batch prediction, long-text prediction, structured decisions, hooks, calibration,
evaluation, router, HTTP/MCP and optional integrations use the upstream host
contracts. MLX owns tensor execution, precision, compilation and prefix caching.
PyTorch/CUDA/ONNX backend switching and training remain explicit native boundaries.

The [public API audit](UPSTREAM_0412_API_AUDIT.json) checks 341 entries, including
original positional order and defaults when a native constructor adds keyword
options. The [progress report](UPSTREAM_0412_PROGRESS.json) records compatibility,
real-model validation, package gates and remaining qualification work. See
[qualification evidence](QUALIFICATION_0_4_12.md) and
[BENCHMARKS.md](../BENCHMARKS.md) for measured workloads and precision limitations.

`benchmarks/qualification.py` runs the sibling Core ML correctness campaign and
then the isolated three-backend FP32 feature matrix. It requires locally exported
adaptive bundles documented in the sibling Core ML conversion guide. It must run
without concurrent accelerator jobs. The final matrix includes alternating input
shapes and records peak process RSS; RSS is not a complete device-allocation meter.

Actual optional-framework smoke tests use LangChain/LangGraph, LlamaIndex, CrewAI and Agno with a deterministic local decision runner and forbidden network access. Both native packages pass all eight integration checks. The LlamaIndex adapter fixes an upstream compatibility bug: current llama-index-core exports SingleSelection rather than ToolSelection and requires prompt-mixin abstract methods. Selectors expose no LLM prompt templates; configure typed instructions directly. Both single and multi selectors now inherit the actual framework BaseSelector. See benchmarks/results/v0412-real-integrations.json for versions and scope. Both dependency lockfiles now include the new extras.

Final local qualification is complete for the pinned upstream and measured workloads. See [the consolidated evidence](QUALIFICATION_0_4_12.md) for exact scope, paired timing, original slower runs, package checks and resource limitations. The migration remains unpublished.
