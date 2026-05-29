"""treeattn -- JAX-backed hierarchical (tree) attention baseline.

Wraps ``treeattn_jax.jax_tree_attention_*`` (``torch.autograd.Function``
shells that bridge into JAX via DLPack). Requires ``jax`` / ``jaxlib``.

Mode selection
--------------

Same convention as :mod:`attn_bench.baselines.treeattn_torch`:

* ``TREEATTN_NUM_SAMPLES`` unset / empty / ``"0"`` ->
  ``jax_tree_attention_{causal,noncausal}_deterministic`` (exact);
* positive integer ``N`` ->
  ``jax_tree_attention_{causal,noncausal}_stochastic`` with ``num_samples=N``.
"""

from __future__ import annotations

import os as _os

# Prevent JAX from preallocating ~75% of GPU memory on first use, which
# would otherwise mask per-adapter peak-memory measurements. Set before
# any `jax` import in the process. `setdefault` lets users override.
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from typing import Callable

import torch

from attn_bench.adapters import KernelAdapter
from attn_bench.baselines._treeattn_common import (
    from_sbhd,
    haar_internal_node_keys,
    is_power_of_two,
    resolve_num_samples,
    to_sbhd,
)


BASELINE_NAME = "treeattn_jax"


def _entrypoints(num_samples: int | None) -> tuple[Callable, Callable]:
    """Return ``(noncausal_fn, causal_fn)`` for the resolved mode."""
    import treeattn_jax as t
    if num_samples is None:
        return (
            t.jax_tree_attention_noncausal_deterministic,
            t.jax_tree_attention_causal_deterministic,
        )
    return (
        t.jax_tree_attention_noncausal_stochastic,
        t.jax_tree_attention_causal_stochastic,
    )


def _call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,  # ignored: tree attention has implicit scaling
) -> torch.Tensor:
    num_samples = resolve_num_samples()
    noncausal_fn, causal_fn = _entrypoints(num_samples)

    B, S, H, D = q.shape
    if not is_power_of_two(S):
        raise ValueError(f"treeattn_jax requires power-of-two seqlen, got {S}")

    q_sbhd = to_sbhd(q)
    k_sbhd = to_sbhd(k)
    v_sbhd = to_sbhd(v)

    k_internal = haar_internal_node_keys(k_sbhd)  # (S-1, B, H, D)

    fn = causal_fn if causal else noncausal_fn
    if num_samples is None:
        # Signature: (q, k, v, dropout, return_weights, max_logit)
        out_sbhd = fn(q_sbhd, k_internal, v_sbhd, None, False, 1e3)
    else:
        # Signature: (q, k, v, num_samples, dropout, return_weights,
        #             gumbel_scale, max_logit)
        out_sbhd = fn(
            q_sbhd, k_internal, v_sbhd,
            num_samples, None, False, 1.0, 1e3,
        )
    # out shape: (L=S, B, H, D)
    return from_sbhd(out_sbhd)


def adapter() -> KernelAdapter:
    return KernelAdapter(
        name=BASELINE_NAME,
        fn=_call,
        supports_backward=True,
        supports_causal=True,
        allowed_dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
        allowed_head_dims=frozenset({32, 64, 96, 128}),
        layout="bshd",
    )


def is_available() -> bool:
    try:
        import treeattn_jax  # noqa: F401
        return True
    except Exception:
        return False
