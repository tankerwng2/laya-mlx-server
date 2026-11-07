# Upstream idle-window timing instability

The public 0.5.0 release passed its full release pipeline. Subsequent documentation
commits exposed intermittent failures in the unchanged upstream
`test_idle_window_starts_after_worker_finishes_even_on_cancel_or_failure` test.
Running host contracts in a separate process before native tests passed one full
CI run, but did not eliminate the failure. No runtime, assertion or timeout was
changed to obtain a pass.

[Direct comparison run](https://github.com/mizorewww/laya-mlx/actions/runs/39355649931):

- Adapted upstream host suite: one idle-window parameter failed; 296 serve tests passed.
- Native/integration suite: 254 passed.
- Original pinned upstream `laya.serve`, using the original `tests/test_serve.py`
  directly with `PYTHONPATH=.upstream`: one idle-window parameter failed, seven passed.

The original and ported `_resolve_idle_unload_seconds`, `_mark_request`,
`_unload_if_idle` and `_idle_reaper` function ASTs match. The test observes a
100 ms no-unload window after a worker completes, with a 200 ms idle threshold;
its wall-clock/thread scheduling assumptions can fail on this runner. The direct
upstream reproduction establishes that this is also present without namespace
adaptation. It does not establish universal scheduler behavior or remove the flake.

The latest documentation CI may remain red for this test. We retain that result;
we do not silently retry until green, skip it, or widen the assertion. CI runs
native tests even if the host suite fails, and runs the original upstream
idle-window tests as failure diagnostics. The published package source is unchanged.
