"""Core timing harness.

* Pins TF32 / cuDNN benchmark off at import time so timings are deterministic
  and "fp32" (when used) means real fp32.
* Times forward and backward passes separately with :class:`torch.cuda.Event`
  pairs, after a warmup phase.
* Catches OOM and arbitrary exceptions per-config so one failure does not
  abort the sweep.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from attn_bench.adapters import (
    KernelAdapter,
    _ptr_snapshot,
    check_no_inplace,
    validate_inputs,
)
from attn_bench.configs import BenchConfig, dtype_str
from attn_bench.flops import bwd_flops, fwd_flops, tflops_per_sec


# Reproducibility: disable TF32 + cuDNN benchmark globally. We do this at
# import time so anyone using the harness picks up the setting even without
# calling a setup function.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("highest")


@dataclass
class TimingStats:
    median_ms: float
    mean_ms: float
    std_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples: list[float]) -> "TimingStats":
        n = len(samples)
        sorted_s = sorted(samples)
        return cls(
            median_ms=statistics.median(sorted_s),
            mean_ms=statistics.fmean(sorted_s),
            std_ms=statistics.pstdev(sorted_s) if n > 1 else 0.0,
            p10_ms=sorted_s[max(0, int(0.10 * (n - 1)))],
            p90_ms=sorted_s[min(n - 1, int(0.90 * (n - 1)))],
            min_ms=sorted_s[0],
            max_ms=sorted_s[-1],
            n=n,
        )


@dataclass
class Result:
    kernel: str
    config: BenchConfig
    status: str  # "ok" | "oom" | "skipped" | "error"
    fwd_inference: Optional[TimingStats] = None
    step: Optional[TimingStats] = None
    fwd_inference_tflops: float = float("nan")
    step_tflops: float = float("nan")
    fwd_peak_mem_mb: float = float("nan")
    fwd_saved_mem_mb: float = float("nan")
    bwd_peak_mem_mb: float = float("nan")
    error_msg: str = ""
    skip_reason: str = ""
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        """Flat dict for CSV/table rendering."""
        cfg = self.config
        row: dict = {
            "kernel": self.kernel,
            "dtype": dtype_str(cfg.dtype),
            "batch": cfg.batch,
            "seqlen": cfg.seqlen,
            "nheads": cfg.nheads,
            "head_dim": cfg.head_dim,
            "causal": cfg.causal,
            "status": self.status,
        }
        if self.fwd_inference is not None:
            row.update({
                "fwd_inference_ms_median": self.fwd_inference.median_ms,
                "fwd_inference_ms_mean": self.fwd_inference.mean_ms,
                "fwd_inference_ms_std": self.fwd_inference.std_ms,
                "fwd_inference_ms_p10": self.fwd_inference.p10_ms,
                "fwd_inference_ms_p90": self.fwd_inference.p90_ms,
                "fwd_inference_tflops": self.fwd_inference_tflops,
            })
        if self.step is not None:
            row.update({
                "step_ms_median": self.step.median_ms,
                "step_ms_mean": self.step.mean_ms,
                "step_ms_std": self.step.std_ms,
                "step_ms_p10": self.step.p10_ms,
                "step_ms_p90": self.step.p90_ms,
                "step_tflops": self.step_tflops,
            })
        if self.status == "ok":
            row.update({
                "fwd_peak_mem_mb": self.fwd_peak_mem_mb,
                "fwd_saved_mem_mb": self.fwd_saved_mem_mb,
                "bwd_peak_mem_mb": self.bwd_peak_mem_mb,
            })
        if self.error_msg:
            row["error_msg"] = self.error_msg
        if self.skip_reason:
            row["skip_reason"] = self.skip_reason
        return row


def _make_inputs(cfg: BenchConfig, *, requires_grad: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    shape = (cfg.batch, cfg.seqlen, cfg.nheads, cfg.head_dim)
    q = torch.randn(shape, device="cuda", dtype=cfg.dtype, generator=gen, requires_grad=requires_grad)
    k = torch.randn(shape, device="cuda", dtype=cfg.dtype, generator=gen, requires_grad=requires_grad)
    v = torch.randn(shape, device="cuda", dtype=cfg.dtype, generator=gen, requires_grad=requires_grad)
    g = torch.randn(shape, device="cuda", dtype=cfg.dtype, generator=gen)
    return q, k, v, g


def _warmup_with_leak_check(
    fn,
    n: int,
    *,
    on_leak,
    threshold_mb: float,
) -> None:
    """Run ``fn()`` ``n`` times as warmup; check for per-iter allocator growth.

    Between every warmup iter we synchronize, ``empty_cache``, and compare
    ``torch.cuda.memory_allocated`` against a baseline established after the
    first iter (the first iter is allowed to allocate persistent workspace).
    If the delta exceeds ``threshold_mb``, ``on_leak(iter_idx, delta_mb)`` is
    called. Always continues; never raises.
    """
    baseline: int | None = None
    for i in range(n):
        fn()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated()
        if baseline is None:
            # First iter establishes the baseline -- some kernels lazily
            # allocate workspaces on first invocation, which is expected.
            baseline = allocated
            continue
        delta_mb = (allocated - baseline) / (1024 * 1024)
        if delta_mb > threshold_mb:
            on_leak(i, delta_mb)
            # Re-baseline so a single persistent growth doesn't fire every iter.
            baseline = allocated


def _time_iters(fn, iters: int) -> list[float]:
    """Time ``fn()`` back-to-back ``iters`` times with CUDA events.

    No cleanup between iterations; the timed loop must be hot-path only so
    measured ms reflect kernel cost rather than allocator/driver overhead.
    """
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _time_callable(fn, *, warmup: int, iters: int) -> list[float]:
    """Backwards-compatible: warm up then time, with no leak checking."""
    for _ in range(warmup):
        fn()
    return _time_iters(fn, iters)


def _jax_bytes_in_use() -> int:
    """Return total bytes_in_use across local JAX devices, or 0.

    Tolerates: jax not installed, PJRT plugins that return ``None`` from
    ``memory_stats()``, and stats dicts missing ``bytes_in_use``.
    """
    try:
        import jax  # type: ignore
    except Exception:
        return 0
    total = 0
    try:
        for d in jax.local_devices():
            stats = d.memory_stats() if hasattr(d, "memory_stats") else None
            if not stats:
                continue
            total += int(stats.get("bytes_in_use", 0))
    except Exception:
        return 0
    return total


def _device_bytes_in_use() -> int:
    """torch CUDA allocator bytes + JAX devices' bytes_in_use, summed.

    Lets memory probes see allocations made through framework-foreign
    allocators (e.g. JAX/XLA's BFC pool) that ``torch.cuda.memory_allocated``
    alone would miss.
    """
    return torch.cuda.memory_allocated() + _jax_bytes_in_use()


def _measure_peak_mem_mb(fn) -> float:
    """Run ``fn()`` once and return an approximate device peak (MiB).

    Combines torch's true peak counter with JAX's resident-bytes delta
    around the call. JAX/PJRT exposes ``bytes_in_use`` (current resident)
    but no peak counter, so the JAX contribution is a lower bound (it
    misses transients that were freed before sync). For pure-torch
    kernels this reduces to ``torch.cuda.max_memory_allocated``.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    jax_before = _jax_bytes_in_use()
    fn()
    torch.cuda.synchronize()
    torch_peak = torch.cuda.max_memory_allocated()
    jax_after = _jax_bytes_in_use()
    jax_delta = max(jax_after - jax_before, 0)
    return (torch_peak + jax_delta) / (1024 * 1024)


def _measure_fwd_saved_mb(fn) -> float:
    """Device memory retained after a forward, with output kept alive.

    Runs ``fn()`` with grad enabled, measures the cross-allocator delta
    while the returned output (and thus saved-for-backward tensors and
    any framework-side VJP closure) is still alive, then drops it.
    """
    torch.cuda.synchronize()
    before = _device_bytes_in_use()
    out = fn()
    torch.cuda.synchronize()
    after = _device_bytes_in_use()
    del out
    return max(after - before, 0) / (1024 * 1024)


def time_kernel(
    adapter: KernelAdapter,
    cfg: BenchConfig,
    *,
    warmup: int = 10,
    iters: int = 50,
    do_backward: bool = True,
    seed: int = 0,
    leak_threshold_mb: float = 1.0,
) -> Result:
    """Time forward (and optionally backward) of one kernel on one config.

    Warmup iterations include an allocator-leak check (warns to stderr); the
    measured ``iters`` loop runs back-to-back without cleanup so timing is
    not contaminated.
    """
    import sys as _sys

    def _on_leak(kind: str):
        def _cb(iter_idx: int, delta_mb: float) -> None:
            print(
                f"[warn] per-iter leak: {adapter.name} s={cfg.seqlen} "
                f"phase={kind} iter={iter_idx} +{delta_mb:.2f} MiB",
                file=_sys.stderr,
            )
        return _cb
    compat, reason = adapter.is_compatible(cfg.dtype, cfg.head_dim, cfg.causal)
    if not compat:
        return Result(
            kernel=adapter.name, config=cfg, status="skipped", skip_reason=reason,
        )

    if not torch.cuda.is_available():
        return Result(
            kernel=adapter.name, config=cfg, status="error",
            error_msg="CUDA not available",
        )

    want_bwd = do_backward and adapter.supports_backward

    try:
        q, k, v, grad_out = _make_inputs(cfg, requires_grad=want_bwd, seed=seed)
        validate_inputs(adapter, q, k, v, cfg.dtype)

        softmax_scale = 1.0 / (cfg.head_dim ** 0.5)
        pre_ptrs = _ptr_snapshot(q, k, v)

        fwd_flops_val = fwd_flops(cfg.batch, cfg.seqlen, cfg.nheads, cfg.head_dim, cfg.causal)
        bwd_flops_val = bwd_flops(cfg.batch, cfg.seqlen, cfg.nheads, cfg.head_dim, cfg.causal)

        # ===== timing pass 1: inference forward (no autograd) =====
        def fwd_inference():
            with torch.no_grad():
                _ = adapter.fn(q, k, v, cfg.causal, softmax_scale)

        _warmup_with_leak_check(
            fwd_inference, warmup,
            on_leak=_on_leak("fwd_inference"),
            threshold_mb=leak_threshold_mb,
        )
        fwd_inf_samples = _time_iters(fwd_inference, iters)
        fwd_inf_stats = TimingStats.from_samples(fwd_inf_samples)
        check_no_inplace(pre_ptrs, (q, k, v))
        fwd_inf_tf = tflops_per_sec(fwd_flops_val, fwd_inf_stats.median_ms)

        # ===== timing pass 2: training step (fwd + bwd) =====
        step_stats: Optional[TimingStats] = None
        step_tf = float("nan")
        if want_bwd:
            def step():
                if q.grad is not None:
                    q.grad = None
                if k.grad is not None:
                    k.grad = None
                if v.grad is not None:
                    v.grad = None
                out = adapter.fn(q, k, v, cfg.causal, softmax_scale)
                out.backward(grad_out)

            _warmup_with_leak_check(
                step, warmup,
                on_leak=_on_leak("step"),
                threshold_mb=leak_threshold_mb,
            )
            step_samples = _time_iters(step, iters)
            step_stats = TimingStats.from_samples(step_samples)
            step_tf = tflops_per_sec(fwd_flops_val + bwd_flops_val, step_stats.median_ms)

        # ===== memory pass (untimed): three independent probes =====
        # Each probe is followed by empty_cache + grad reset to keep the
        # next measurement isolated.
        def _reset_grads():
            if q.grad is not None: q.grad = None
            if k.grad is not None: k.grad = None
            if v.grad is not None: v.grad = None

        # (a) forward peak with grad enabled (training-forward memory).
        def fwd_with_grad_no_keep():
            _ = adapter.fn(q, k, v, cfg.causal, softmax_scale)
        if want_bwd:
            fwd_peak_mb = _measure_peak_mem_mb(fwd_with_grad_no_keep)
        else:
            fwd_peak_mb = _measure_peak_mem_mb(fwd_inference)
        torch.cuda.empty_cache()
        _reset_grads()

        # (b) memory retained for backward: forward with grad on, output kept
        #     alive across the snapshot so its autograd graph + saved tensors
        #     (including framework-foreign VJP closures) count.
        if want_bwd:
            def fwd_returning_with_grad():
                return adapter.fn(q, k, v, cfg.causal, softmax_scale)
            fwd_saved_mb = _measure_fwd_saved_mb(fwd_returning_with_grad)
        else:
            fwd_saved_mb = float("nan")
        torch.cuda.empty_cache()
        _reset_grads()

        # (c) full fwd+bwd peak.
        if want_bwd:
            def step_for_mem():
                _reset_grads()
                out = adapter.fn(q, k, v, cfg.causal, softmax_scale)
                out.backward(grad_out)
            bwd_peak_mb = _measure_peak_mem_mb(step_for_mem)
        else:
            bwd_peak_mb = float("nan")
        torch.cuda.empty_cache()
        _reset_grads()

        return Result(
            kernel=adapter.name,
            config=cfg,
            status="ok",
            fwd_inference=fwd_inf_stats,
            step=step_stats,
            fwd_inference_tflops=fwd_inf_tf,
            step_tflops=step_tf,
            fwd_peak_mem_mb=fwd_peak_mb,
            fwd_saved_mem_mb=fwd_saved_mb,
            bwd_peak_mem_mb=bwd_peak_mb,
        )

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return Result(
            kernel=adapter.name, config=cfg, status="oom", error_msg=str(e),
        )
    except Exception as e:
        # Defensive: any unexpected error becomes a per-cell failure, not a
        # whole-sweep crash.
        return Result(
            kernel=adapter.name, config=cfg, status="error",
            error_msg=f"{type(e).__name__}: {e}",
        )
    finally:
        # Help the next config avoid fragmentation-related OOMs.
        try:
            del q, k, v, grad_out  # type: ignore[possibly-unbound]
        except NameError:
            pass
        torch.cuda.empty_cache()


def run_sweep(
    adapters: list[KernelAdapter],
    sweep: list[BenchConfig],
    *,
    warmup: int = 10,
    iters: int = 50,
    do_backward: bool = True,
    seed: int = 0,
    on_result=None,
    leak_threshold_mb: float = 1.0,
) -> list[Result]:
    """Run every (adapter, config) pair. Calls ``on_result`` after each cell.

    Loop order is **adapter -> config** (outer to inner) so each adapter
    completes its full sweep before the next one starts. Between cells,
    drops references and calls ``torch.cuda.empty_cache``. If the allocator
    still holds more than ``leak_threshold_mb`` MiB more than it did before
    the cell started, prints a warning identifying the offending pair.

    OOM short-circuit: once an adapter hits ``status="oom"`` at some
    seqlen, every later config with a larger seqlen is skipped for that
    adapter (with ``status="skipped"`` and a descriptive ``skip_reason``).
    The short-circuit resets per adapter.
    """
    import sys as _sys
    results: list[Result] = []
    for adapter in adapters:
        oom_at_seqlen: int | None = None
        for cfg in sweep:
            if oom_at_seqlen is not None and cfg.seqlen > oom_at_seqlen:
                res = Result(
                    kernel=adapter.name, config=cfg, status="skipped",
                    skip_reason=f"prior OOM at s={oom_at_seqlen}",
                )
                results.append(res)
                if on_result is not None:
                    on_result(res)
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated()
            else:
                mem_before = 0

            t0 = time.perf_counter()
            res = time_kernel(
                adapter, cfg, warmup=warmup, iters=iters,
                do_backward=do_backward, seed=seed,
                leak_threshold_mb=leak_threshold_mb,
            )
            res.extra["wallclock_s"] = time.perf_counter() - t0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                mem_after = torch.cuda.memory_allocated()
                leaked_mb = (mem_after - mem_before) / (1024 * 1024)
                res.extra["leaked_mb"] = leaked_mb
                if leaked_mb > leak_threshold_mb:
                    print(
                        f"[warn] leak: {adapter.name} s={cfg.seqlen} retained "
                        f"{leaked_mb:.1f} MiB after empty_cache",
                        file=_sys.stderr,
                    )

            if res.status == "oom":
                oom_at_seqlen = cfg.seqlen

            results.append(res)
            if on_result is not None:
                on_result(res)
    return results


def time_kernel_by_name(
    name: str,
    cfg: BenchConfig,
    **kwargs,
) -> Result:
    """Instantiate a built-in baseline by name and call :func:`time_kernel`."""
    from attn_bench.baselines import get_baseline
    return time_kernel(get_baseline(name), cfg, **kwargs)


# --- subprocess-isolated runner ----------------------------------------------
#
# Each adapter runs in its own ``multiprocessing`` child so that CUDA context,
# framework allocators (PyTorch caching allocator, JAX/XLA BFC pool), and any
# kernel-internal workspaces are torn down completely between adapters.
# Children stream :class:`Result` objects back via a ``Queue``; the parent
# dispatches them through ``on_result`` exactly as the in-process runner does.


def _adapter_child(
    name: str,
    sweep: list[BenchConfig],
    kwargs: dict,
    queue,
) -> None:
    """Worker entrypoint: run the full sweep for one adapter, stream results.

    Module-level (picklable) so :mod:`multiprocessing` ``spawn`` can import
    and call it. Always puts a final ``None`` sentinel so the parent can
    detect orderly completion vs. crash (no sentinel + nonzero exit).
    """
    import os
    import sys as _sys

    # Re-assert in case the child was spawned via a context that did not
    # inherit the parent's modifications (spawn copies os.environ, so this
    # is belt-and-suspenders).
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    try:
        from attn_bench.baselines import get_baseline
        adapter = get_baseline(name)
    except Exception as e:
        # Couldn't even build the adapter -- mark every cfg as error.
        for cfg in sweep:
            queue.put(Result(
                kernel=name, config=cfg, status="error",
                error_msg=f"adapter init failed: {type(e).__name__}: {e}",
            ))
        queue.put(None)
        return

    warmup = kwargs.get("warmup", 10)
    iters = kwargs.get("iters", 50)
    do_backward = kwargs.get("do_backward", True)
    seed = kwargs.get("seed", 0)
    leak_threshold_mb = kwargs.get("leak_threshold_mb", 1.0)

    oom_at_seqlen: int | None = None
    for cfg in sweep:
        if oom_at_seqlen is not None and cfg.seqlen > oom_at_seqlen:
            queue.put(Result(
                kernel=name, config=cfg, status="skipped",
                skip_reason=f"prior OOM at s={oom_at_seqlen}",
            ))
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
        else:
            mem_before = 0

        t0 = time.perf_counter()
        try:
            res = time_kernel(
                adapter, cfg,
                warmup=warmup, iters=iters,
                do_backward=do_backward, seed=seed,
                leak_threshold_mb=leak_threshold_mb,
            )
        except Exception as e:
            res = Result(
                kernel=name, config=cfg, status="error",
                error_msg=f"{type(e).__name__}: {e}",
            )
        res.extra["wallclock_s"] = time.perf_counter() - t0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated()
            leaked_mb = (mem_after - mem_before) / (1024 * 1024)
            res.extra["leaked_mb"] = leaked_mb
            if leaked_mb > leak_threshold_mb:
                print(
                    f"[warn] leak: {name} s={cfg.seqlen} retained "
                    f"{leaked_mb:.1f} MiB after empty_cache",
                    file=_sys.stderr,
                )

        if res.status == "oom":
            oom_at_seqlen = cfg.seqlen

        queue.put(res)

    queue.put(None)


def run_sweep_isolated(
    names: list[str],
    sweep: list[BenchConfig],
    *,
    warmup: int = 10,
    iters: int = 50,
    do_backward: bool = True,
    seed: int = 0,
    on_result=None,
    leak_threshold_mb: float = 1.0,
) -> list[Result]:
    """Run each adapter in its own subprocess, in series.

    The parent never touches the adapter modules; only the child imports
    torch-side baselines and (for JAX) initializes XLA. When the child exits,
    its CUDA context is destroyed and all GPU memory (PyTorch caching
    allocator + JAX BFC pool + any kernel-internal pools) is released back
    to the driver.
    """
    import multiprocessing as mp
    import sys as _sys

    ctx = mp.get_context("spawn")
    kwargs = {
        "warmup": warmup,
        "iters": iters,
        "do_backward": do_backward,
        "seed": seed,
        "leak_threshold_mb": leak_threshold_mb,
    }

    results: list[Result] = []
    for name in names:
        queue = ctx.Queue()
        proc = ctx.Process(target=_adapter_child, args=(name, sweep, kwargs, queue))
        proc.start()

        reported_for_adapter: list[Result] = []
        saw_sentinel = False
        while True:
            try:
                # Long timeout to catch hung children without busy-waiting.
                msg = queue.get(timeout=3600)
            except Exception:
                break
            if msg is None:
                saw_sentinel = True
                break
            reported_for_adapter.append(msg)
            results.append(msg)
            if on_result is not None:
                on_result(msg)

        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)

        if not saw_sentinel:
            # Child crashed or was killed before finishing the sweep.
            reported_cfgs = {(r.config.seqlen, r.config.batch, r.config.nheads,
                              r.config.head_dim, r.config.causal)
                             for r in reported_for_adapter}
            for cfg in sweep:
                key = (cfg.seqlen, cfg.batch, cfg.nheads, cfg.head_dim, cfg.causal)
                if key in reported_cfgs:
                    continue
                synth = Result(
                    kernel=name, config=cfg, status="error",
                    error_msg=f"child process exited prematurely "
                              f"(exitcode={proc.exitcode})",
                )
                results.append(synth)
                if on_result is not None:
                    on_result(synth)
            print(
                f"[warn] adapter {name!r} subprocess exited with code "
                f"{proc.exitcode} before completing sweep",
                file=_sys.stderr,
            )

    return results
