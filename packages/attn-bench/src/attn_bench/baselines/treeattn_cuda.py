"""treeattn -- staged CUDA baseline.

This baseline exposes the first implementation slice of a direct PyTorch
autograd path that is intended to be replaced by native CUDA kernels.
Currently only non-causal stochastic mode is custom; other modes fall back to
the reference tree attention package and are therefore not benchmarked here.
"""

from __future__ import annotations

from attn_bench.adapters import KernelAdapter
from attn_bench.baselines._treeattn_common import (
    from_sbhd,
    haar_internal_node_keys,
    is_power_of_two,
    resolve_num_samples,
    to_sbhd,
)


BASELINE_NAME = "treeattn_cuda"


def _call(q, k, v, causal, softmax_scale):
    del softmax_scale
    if causal:
        raise ValueError("treeattn_cuda does not support causal attention yet")

    num_samples = resolve_num_samples()
    if num_samples is None:
        raise ValueError(
            "treeattn_cuda currently requires TREEATTN_NUM_SAMPLES to be set"
        )

    import treeattn_cuda as t

    batch, seqlen, nheads, width = q.shape
    if not is_power_of_two(seqlen):
        raise ValueError(f"treeattn_cuda requires power-of-two seqlen, got {seqlen}")

    q_sbhd = to_sbhd(q)
    k_sbhd = to_sbhd(k)
    v_sbhd = to_sbhd(v)
    k_internal = haar_internal_node_keys(k_sbhd)

    q_lbe = q_sbhd.reshape(seqlen, batch, nheads * width)
    k_lbe = k_internal.reshape(seqlen - 1, batch, nheads * width)
    v_lbe = v_sbhd.reshape(seqlen, batch, nheads * width)
    out_lbe, _ = t.hierarchical_attention(
        q_lbe,
        k_lbe,
        v_lbe,
        num_heads=nheads,
        is_causal=False,
        mode="stochastic",
        num_samples=num_samples,
    )
    out_sbhd = out_lbe.reshape(seqlen, batch, nheads, width)
    return from_sbhd(out_sbhd)


def adapter() -> KernelAdapter:
    import torch

    return KernelAdapter(
        name=BASELINE_NAME,
        fn=_call,
        supports_backward=True,
        supports_causal=False,
        allowed_dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
        allowed_head_dims=frozenset({32, 64, 96, 128}),
        layout="bshd",
    )


def is_available() -> bool:
    try:
        import treeattn_cuda  # noqa: F401

        return True
    except Exception:
        return False