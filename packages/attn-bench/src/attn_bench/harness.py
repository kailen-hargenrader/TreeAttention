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
    fwd: Optional[TimingStats] = None
    bwd: Optional[TimingStats] = None
    fwd_tflops: float = float("nan")
    bwd_tflops: float = float("nan")
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
        if self.fwd is not None:
            row.update({
                "fwd_ms_median": self.fwd.median_ms,
                "fwd_ms_mean": self.fwd.mean_ms,
                "fwd_ms_std": self.fwd.std_ms,
                "fwd_ms_p10": self.fwd.p10_ms,
                "fwd_ms_p90": self.fwd.p90_ms,
                "fwd_tflops": self.fwd_tflops,
            })
        if self.bwd is not None:
            row.update({
                "bwd_ms_median": self.bwd.median_ms,
                "bwd_ms_mean": self.bwd.mean_ms,
                "bwd_ms_std": self.bwd.std_ms,
                "bwd_ms_p10": self.bwd.p10_ms,
                "bwd_ms_p90": self.bwd.p90_ms,
                "bwd_tflops": self.bwd_tflops,
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


def _time_callable(fn, *, warmup: int, iters: int) -> list[float]:
    """Time ``fn()`` ``iters`` times after ``warmup`` warmup calls.

    ``fn`` must perform the work but not call ``cuda.synchronize`` itself.
    Returns a list of per-iteration elapsed times in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def time_kernel(
    adapter: KernelAdapter,
    cfg: BenchConfig,
    *,
    warmup: int = 10,
    iters: int = 50,
    do_backward: bool = True,
    seed: int = 0,
) -> Result:
    """Time forward (and optionally backward) of one kernel on one config."""
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

        # ----- forward timing -----
        def fwd_only():
            with torch.no_grad():
                _ = adapter.fn(q, k, v, cfg.causal, softmax_scale)

        fwd_samples = _time_callable(fwd_only, warmup=warmup, iters=iters)
        fwd_stats = TimingStats.from_samples(fwd_samples)
        check_no_inplace(pre_ptrs, (q, k, v))

        fwd_flops_val = fwd_flops(cfg.batch, cfg.seqlen, cfg.nheads, cfg.head_dim, cfg.causal)
        fwd_tf = tflops_per_sec(fwd_flops_val, fwd_stats.median_ms)

        bwd_stats: Optional[TimingStats] = None
        bwd_tf = float("nan")

        if want_bwd:
            # Rebuild inputs with grads so each backward call has fresh leaves.
            # We pay the construction cost once and zero grads between iters.
            def bwd_step():
                if q.grad is not None:
                    q.grad = None
                if k.grad is not None:
                    k.grad = None
                if v.grad is not None:
                    v.grad = None
                out = adapter.fn(q, k, v, cfg.causal, softmax_scale)
                out.backward(grad_out)

            # We can't separately time backward without re-running forward
            # (since the autograd graph is consumed). Common practice: time
            # fwd+bwd together, then subtract median forward. We instead time
            # full step ("fwd+bwd") and report bwd_ms = step_ms - fwd_ms; this
            # matches the FA paper's reporting.
            step_samples = _time_callable(bwd_step, warmup=warmup, iters=iters)
            step_stats = TimingStats.from_samples(step_samples)
            # Per-iter subtraction would be noisier; use median of step minus
            # median of fwd, clamped at 0.
            bwd_median = max(step_stats.median_ms - fwd_stats.median_ms, 0.0)
            # For std/percentiles, fall back to step stats minus fwd median.
            bwd_samples = [max(s - fwd_stats.median_ms, 0.0) for s in step_samples]
            bwd_stats = TimingStats.from_samples(bwd_samples)
            bwd_stats.median_ms = bwd_median  # type: ignore[misc]

            bwd_flops_val = bwd_flops(cfg.batch, cfg.seqlen, cfg.nheads, cfg.head_dim, cfg.causal)
            bwd_tf = tflops_per_sec(bwd_flops_val, bwd_stats.median_ms)

        return Result(
            kernel=adapter.name,
            config=cfg,
            status="ok",
            fwd=fwd_stats,
            bwd=bwd_stats,
            fwd_tflops=fwd_tf,
            bwd_tflops=bwd_tf,
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
) -> list[Result]:
    """Run every (adapter, config) pair. Calls ``on_result`` after each cell."""
    results: list[Result] = []
    for cfg in sweep:
        for adapter in adapters:
            t0 = time.perf_counter()
            res = time_kernel(
                adapter, cfg, warmup=warmup, iters=iters,
                do_backward=do_backward, seed=seed,
            )
            res.extra["wallclock_s"] = time.perf_counter() - t0
            results.append(res)
            if on_result is not None:
                on_result(res)
    return results
