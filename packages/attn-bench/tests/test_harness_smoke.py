"""Smoke test: run a single tiny config through the harness against FA2.

Skipped (rather than failed) when CUDA or flash-attn is unavailable, so the
test passes on environments without GPUs while still exercising the full
code path when run on A100.
"""

from __future__ import annotations

import math

import pytest
import torch

from attn_bench.baselines import sdpa
from attn_bench.baselines.flash_attn2 import adapter as fa2_adapter, is_available as fa2_available
from attn_bench.configs import BenchConfig
from attn_bench.harness import time_kernel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _tiny_cfg() -> BenchConfig:
    return BenchConfig(
        batch=1, seqlen=128, nheads=2, head_dim=64,
        dtype=torch.bfloat16, causal=False,
    )


def test_fa2_smoke() -> None:
    if not fa2_available():
        pytest.skip("flash-attn not installed")
    cfg = _tiny_cfg()
    res = time_kernel(fa2_adapter(), cfg, warmup=2, iters=5)
    assert res.status == "ok", res.error_msg or res.skip_reason
    assert res.fwd is not None and math.isfinite(res.fwd.median_ms)
    assert res.bwd is not None and math.isfinite(res.bwd.median_ms)
    assert res.fwd.median_ms > 0
    assert res.bwd.median_ms >= 0


def test_sdpa_smoke() -> None:
    cfg = _tiny_cfg()
    res = time_kernel(sdpa.adapter(), cfg, warmup=2, iters=5)
    assert res.status == "ok", res.error_msg or res.skip_reason
    assert res.fwd is not None and math.isfinite(res.fwd.median_ms)
    assert res.fwd.median_ms > 0


def test_peak_memory_reported() -> None:
    cfg = _tiny_cfg()
    res = time_kernel(sdpa.adapter(), cfg, warmup=2, iters=5)
    assert res.status == "ok", res.error_msg or res.skip_reason
    assert math.isfinite(res.fwd_peak_mem_mb)
    assert res.fwd_peak_mem_mb > 0
    assert math.isfinite(res.bwd_peak_mem_mb)
    # fwd+bwd retains activations and produces grads, so should use at
    # least as much memory as a pure forward.
    assert res.bwd_peak_mem_mb >= res.fwd_peak_mem_mb
    row = res.row()
    assert "fwd_peak_mem_mb" in row
    assert "bwd_peak_mem_mb" in row
