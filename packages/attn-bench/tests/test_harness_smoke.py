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
from attn_bench.baselines.treeattn_cuda import adapter as treeattn_cuda_adapter
from attn_bench.baselines.flash_attn2 import adapter as fa2_adapter, is_available as fa2_available
from attn_bench.configs import BenchConfig
from attn_bench.harness import run_sweep_isolated, time_kernel
from treeattn_cuda import hierarchical_attention as cuda_hierarchical_attention
from treeattn_cuda import _native as treeattn_cuda_native
from treeattn_cuda._autograd import (
    _accumulate_qk_non_causal_all_depths_inplace,
    _accumulate_qk_non_causal_inplace,
    _compute_grad_logit_non_causal,
    _replay_non_causal_paths,
    _replay_non_causal_paths_python,
    _scatter_tree_updates,
    _scatter_weighted_grad_v,
    _sample_paths_non_causal_streaming_python,
    _sample_paths_non_causal_streaming,
)
from treeattn_torch import hierarchical_attention as torch_hierarchical_attention


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
    assert res.fwd_inference is not None and math.isfinite(res.fwd_inference.median_ms)
    assert res.step is not None and math.isfinite(res.step.median_ms)
    assert res.fwd_inference.median_ms > 0
    assert res.step.median_ms >= 0


def test_sdpa_smoke() -> None:
    cfg = _tiny_cfg()
    res = time_kernel(sdpa.adapter(), cfg, warmup=2, iters=5)
    assert res.status == "ok", res.error_msg or res.skip_reason
    assert res.fwd_inference is not None and math.isfinite(res.fwd_inference.median_ms)
    assert res.fwd_inference.median_ms > 0


def test_peak_memory_reported() -> None:
    cfg = _tiny_cfg()
    res = time_kernel(sdpa.adapter(), cfg, warmup=2, iters=5)
    assert res.status == "ok", res.error_msg or res.skip_reason
    assert math.isfinite(res.fwd_peak_mem_mb)
    assert res.fwd_peak_mem_mb > 0
    assert math.isfinite(res.fwd_saved_mem_mb)
    assert res.fwd_saved_mem_mb >= 0
    assert math.isfinite(res.bwd_peak_mem_mb)
    # fwd+bwd retains activations and produces grads, so should use at
    # least as much memory as a pure forward.
    assert res.bwd_peak_mem_mb >= res.fwd_peak_mem_mb
    row = res.row()
    assert "fwd_peak_mem_mb" in row
    assert "fwd_saved_mem_mb" in row
    assert "bwd_peak_mem_mb" in row
    assert "fwd_residual_mem_mb" not in row


def test_treeattn_cuda_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _tiny_cfg()
    monkeypatch.setenv("TREEATTN_NUM_SAMPLES", "8")
    res = time_kernel(treeattn_cuda_adapter(), cfg, warmup=1, iters=2)
    assert res.status == "ok", res.error_msg or res.skip_reason
    assert res.fwd_inference is not None and math.isfinite(res.fwd_inference.median_ms)
    assert res.step is not None and math.isfinite(res.step.median_ms)
    assert res.fwd_inference.median_ms > 0


def test_treeattn_cuda_path_replay_roundtrip() -> None:
    torch.manual_seed(0)
    q = torch.randn((8, 1, 2, 16), device="cuda", dtype=torch.float32)
    k = torch.randn((7, 1, 2, 16), device="cuda", dtype=torch.float32)

    sampled_indices, path_log_probs, packed_paths = _sample_paths_non_causal_streaming(
        q,
        k,
        num_samples=4,
        block_size=4,
        max_logit=1e3,
    )
    replayed_indices, replayed_log_probs = _replay_non_causal_paths(
        q,
        k,
        packed_paths,
        block_size=4,
        max_logit=1e3,
    )

    assert packed_paths.dtype == torch.uint8
    assert packed_paths.numel() * packed_paths.element_size() < (
        sampled_indices.numel() * sampled_indices.element_size()
    )
    assert torch.equal(replayed_indices, sampled_indices)
    torch.testing.assert_close(replayed_log_probs, path_log_probs)


def test_treeattn_cuda_native_replay_matches_python() -> None:
    if not treeattn_cuda_native.has_native_kernels():
        pytest.skip("treeattn_cuda native replay kernel not enabled")

    torch.manual_seed(0)
    q = torch.randn((8, 1, 2, 16), device="cuda", dtype=torch.float32)
    k = torch.randn((7, 1, 2, 16), device="cuda", dtype=torch.float32)
    _, _, packed_paths = _sample_paths_non_causal_streaming(
        q,
        k,
        num_samples=4,
        block_size=4,
        max_logit=1e3,
    )

    native_indices, native_log_probs = _replay_non_causal_paths(
        q,
        k,
        packed_paths,
        block_size=4,
        max_logit=1e3,
    )
    python_indices, python_log_probs = _replay_non_causal_paths_python(
        q,
        k,
        packed_paths,
        block_size=4,
        max_logit=1e3,
    )

    assert torch.equal(native_indices, python_indices)
    torch.testing.assert_close(native_log_probs, python_log_probs)


def test_treeattn_cuda_native_sampler_matches_python() -> None:
    if not treeattn_cuda_native.has_native_kernels():
        pytest.skip("treeattn_cuda native sampler kernel not enabled")

    torch.manual_seed(0)
    q = torch.randn((8, 1, 2, 16), device="cuda", dtype=torch.float32)
    k = torch.randn((7, 1, 2, 16), device="cuda", dtype=torch.float32)

    torch.manual_seed(123)
    native_indices, native_log_probs, native_packed = _sample_paths_non_causal_streaming(
        q,
        k,
        num_samples=4,
        block_size=4,
        max_logit=1e3,
    )
    torch.manual_seed(123)
    python_indices, python_log_probs, python_packed = _sample_paths_non_causal_streaming_python(
        q,
        k,
        num_samples=4,
        block_size=4,
        max_logit=1e3,
    )

    assert torch.equal(native_indices, python_indices)
    assert torch.equal(native_packed, python_packed)
    torch.testing.assert_close(native_log_probs, python_log_probs)


def test_treeattn_cuda_native_grad_v_matches_python() -> None:
    if not treeattn_cuda_native.has_native_kernels():
        pytest.skip("treeattn_cuda native grad_v kernel not enabled")

    torch.manual_seed(0)
    grad_output = torch.randn((8, 1, 2, 16), device="cuda", dtype=torch.float32)
    sampled_indices = torch.randint(0, 8, (8, 1, 2, 4), device="cuda", dtype=torch.int64)
    attn_weights = torch.softmax(
        torch.randn((8, 1, 2, 4), device="cuda", dtype=torch.float32),
        dim=-1,
    )

    native_grad_v = _scatter_weighted_grad_v(
        grad_output,
        sampled_indices,
        attn_weights,
        num_leaves=8,
    )
    assert native_grad_v is not None

    python_updates = attn_weights.unsqueeze(-1) * grad_output.unsqueeze(-2)
    python_grad_v = _scatter_tree_updates(8, sampled_indices, python_updates)
    torch.testing.assert_close(native_grad_v, python_grad_v)


def test_treeattn_cuda_native_grad_logit_matches_python() -> None:
    if not treeattn_cuda_native.has_native_kernels():
        pytest.skip("treeattn_cuda native grad_logit kernel not enabled")

    torch.manual_seed(0)
    q = torch.randn((8, 1, 2, 16), device="cuda", dtype=torch.float32)
    k = torch.randn((7, 1, 2, 16), device="cuda", dtype=torch.float32)
    grad_log_probs = torch.randn((8, 1, 2, 4), device="cuda", dtype=torch.float32)
    current_nodes = torch.randint(0, 7, (8, 1, 2, 4), device="cuda", dtype=torch.int64)
    _, _, packed_paths = _sample_paths_non_causal_streaming_python(
        q,
        k,
        num_samples=4,
        block_size=4,
        max_logit=1e3,
    )
    depth = 1

    native_grad_logit = _compute_grad_logit_non_causal(
        q,
        k,
        current_nodes,
        packed_paths,
        grad_log_probs,
        depth=depth,
        max_logit=1e3,
    )
    assert native_grad_logit is not None

    bit = ((packed_paths[..., depth // 8] >> (depth % 8)) & 1).to(torch.int64)
    branch_sign = torch.where(bit == 0, 1.0, -1.0).to(torch.float32)
    gathered_keys = torch.gather(
        k.permute(1, 2, 0, 3).contiguous().expand(8, 1, 2, 7, 16),
        dim=-2,
        index=current_nodes.unsqueeze(-1).expand(8, 1, 2, 4, 16),
    )
    logits = (q.unsqueeze(-2) * gathered_keys).sum(dim=-1).clamp(min=-1e3, max=1e3)
    python_grad_logit = branch_sign * torch.sigmoid(-branch_sign * logits) * grad_log_probs
    torch.testing.assert_close(native_grad_logit, python_grad_logit)


def test_treeattn_cuda_native_qk_accum_matches_python() -> None:
    if not treeattn_cuda_native.has_native_kernels():
        pytest.skip("treeattn_cuda native qk kernel not enabled")

    torch.manual_seed(0)
    q = torch.randn((8, 1, 2, 16), device="cuda", dtype=torch.float32)
    k = torch.randn((7, 1, 2, 16), device="cuda", dtype=torch.float32)
    grad_log_probs = torch.randn((8, 1, 2, 4), device="cuda", dtype=torch.float32)
    current_nodes = torch.randint(0, 7, (8, 1, 2, 4), device="cuda", dtype=torch.int64)
    _, _, packed_paths = _sample_paths_non_causal_streaming_python(
        q,
        k,
        num_samples=4,
        block_size=4,
        max_logit=1e3,
    )
    depth = 1
    grad_q_native = torch.zeros_like(q)
    grad_k_native = torch.zeros_like(k)
    used_native = _accumulate_qk_non_causal_inplace(
        q,
        k,
        current_nodes,
        packed_paths,
        grad_log_probs,
        depth=depth,
        max_logit=1e3,
        grad_q_out=grad_q_native,
        grad_k_out=grad_k_native,
    )
    assert used_native

    bit = ((packed_paths[..., depth // 8] >> (depth % 8)) & 1).to(torch.int64)
    branch_sign = torch.where(bit == 0, 1.0, -1.0).to(torch.float32)
    gathered_keys = torch.gather(
        k.permute(1, 2, 0, 3).contiguous().expand(8, 1, 2, 7, 16),
        dim=-2,
        index=current_nodes.unsqueeze(-1).expand(8, 1, 2, 4, 16),
    )
    logits = (q.unsqueeze(-2) * gathered_keys).sum(dim=-1).clamp(min=-1e3, max=1e3)
    grad_logit = branch_sign * torch.sigmoid(-branch_sign * logits) * grad_log_probs
    grad_q_python = (grad_logit.unsqueeze(-1) * gathered_keys).sum(dim=-2)
    grad_k_updates = grad_logit.unsqueeze(-1) * q.unsqueeze(-2)
    grad_k_python = _scatter_tree_updates(7, current_nodes, grad_k_updates)

    torch.testing.assert_close(grad_q_native, grad_q_python)
    torch.testing.assert_close(grad_k_native, grad_k_python)


def test_treeattn_cuda_native_full_qk_accum_matches_python() -> None:
    if not treeattn_cuda_native.has_native_kernels():
        pytest.skip("treeattn_cuda native fused qk kernel not enabled")

    torch.manual_seed(0)
    q = torch.randn((8, 1, 2, 16), device="cuda", dtype=torch.float32)
    k = torch.randn((7, 1, 2, 16), device="cuda", dtype=torch.float32)
    grad_log_probs = torch.randn((8, 1, 2, 4), device="cuda", dtype=torch.float32)
    _, _, packed_paths = _sample_paths_non_causal_streaming_python(
        q,
        k,
        num_samples=4,
        block_size=4,
        max_logit=1e3,
    )
    grad_q_native = torch.zeros_like(q)
    grad_k_native = torch.zeros_like(k)
    used_native = _accumulate_qk_non_causal_all_depths_inplace(
        q,
        k,
        packed_paths,
        grad_log_probs,
        max_logit=1e3,
        grad_q_out=grad_q_native,
        grad_k_out=grad_k_native,
    )
    assert used_native

    grad_q_python = torch.zeros_like(q)
    grad_k_python = torch.zeros_like(k)
    node_idx = torch.zeros((8, 1, 2, 4), device="cuda", dtype=torch.int64)
    log_n = (k.shape[0] + 1).bit_length() - 1
    for depth in range(log_n):
        bit = ((packed_paths[..., depth // 8] >> (depth % 8)) & 1).to(torch.int64)
        branch_sign = torch.where(bit == 0, 1.0, -1.0).to(torch.float32)
        gathered_keys = torch.gather(
            k.permute(1, 2, 0, 3).contiguous().expand(8, 1, 2, 7, 16),
            dim=-2,
            index=node_idx.unsqueeze(-1).expand(8, 1, 2, 4, 16),
        )
        logits = (q.unsqueeze(-2) * gathered_keys).sum(dim=-1).clamp(min=-1e3, max=1e3)
        grad_logit = branch_sign * torch.sigmoid(-branch_sign * logits) * grad_log_probs
        grad_q_python = grad_q_python + (grad_logit.unsqueeze(-1) * gathered_keys).sum(dim=-2)
        grad_k_updates = grad_logit.unsqueeze(-1) * q.unsqueeze(-2)
        grad_k_python = grad_k_python + _scatter_tree_updates(7, node_idx, grad_k_updates)
        if depth + 1 < log_n:
            node_idx = 2 * node_idx + 1 + bit

    torch.testing.assert_close(grad_q_native, grad_q_python)
    torch.testing.assert_close(grad_k_native, grad_k_python)


def _assert_treeattn_cuda_matches_torch_reference(
    *,
    seqlen: int,
    batch: int,
    nheads: int,
    head_dim: int,
    num_samples: int,
) -> None:
    embed = nheads * head_dim

    def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            (seqlen, batch, embed),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        k = torch.randn(
            (seqlen - 1, batch, embed),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        v = torch.randn(
            (seqlen, batch, embed),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        return q, k, v

    torch.manual_seed(1234)
    q_cuda, k_cuda, v_cuda = _make_inputs()
    q_ref = q_cuda.detach().clone().requires_grad_(True)
    k_ref = k_cuda.detach().clone().requires_grad_(True)
    v_ref = v_cuda.detach().clone().requires_grad_(True)
    grad_out = torch.randn((seqlen, batch, embed), device="cuda", dtype=torch.float32)

    torch.manual_seed(0)
    out_cuda, weights_cuda = cuda_hierarchical_attention(
        q_cuda,
        k_cuda,
        v_cuda,
        num_heads=nheads,
        is_causal=False,
        mode="stochastic",
        num_samples=num_samples,
        block_size=min(256, seqlen),
    )
    out_cuda.backward(grad_out)

    torch.manual_seed(0)
    out_ref, weights_ref = torch_hierarchical_attention(
        q_ref,
        k_ref,
        v_ref,
        num_heads=nheads,
        is_causal=False,
        mode="stochastic",
        num_samples=num_samples,
    )
    out_ref.backward(grad_out)

    torch.testing.assert_close(out_cuda, out_ref, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(weights_cuda, weights_ref, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(q_cuda.grad, q_ref.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(k_cuda.grad, k_ref.grad, atol=4e-4, rtol=4e-4)
    torch.testing.assert_close(v_cuda.grad, v_ref.grad, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    ("seqlen", "batch", "nheads", "head_dim", "num_samples"),
    [
        (8, 1, 2, 16, 4),
        (32, 2, 2, 16, 4),
        (64, 1, 8, 16, 4),
        (128, 1, 2, 64, 8),
        (512, 2, 4, 32, 8),
    ],
)
def test_treeattn_cuda_matches_torch_reference(
    seqlen: int,
    batch: int,
    nheads: int,
    head_dim: int,
    num_samples: int,
) -> None:
    _assert_treeattn_cuda_matches_torch_reference(
        seqlen=seqlen,
        batch=batch,
        nheads=nheads,
        head_dim=head_dim,
        num_samples=num_samples,
    )


def test_isolated_runner_smoke() -> None:
    """One adapter, one tiny config, via subprocess isolation."""
    cfg = _tiny_cfg()
    results = run_sweep_isolated(
        ["sdpa_efficient"], [cfg], warmup=2, iters=3, do_backward=False,
    )
    assert len(results) == 1
    res = results[0]
    assert res.status == "ok", res.error_msg or res.skip_reason
    assert res.kernel == "sdpa_efficient"
    assert res.fwd_inference is not None and math.isfinite(res.fwd_inference.median_ms)
    assert res.fwd_inference.median_ms > 0
