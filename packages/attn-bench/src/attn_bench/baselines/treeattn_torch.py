"""treeattn -- pure-PyTorch hierarchical (tree) attention baseline.

Wraps :func:`treeattn_torch.hierarchical_attention` (vendored from
``tachyon.modules.tree_mha`` with the stochastic-mode K-axis gather bug
fixed; see ``packages/treeattn-torch/README.md``).

Mode selection
--------------

The mode (deterministic vs. stochastic) and the stochastic sample count are
read from the ``TREEATTN_NUM_SAMPLES`` environment variable each time the
kernel is invoked:

* unset / empty / ``"0"`` -> ``mode="deterministic"`` (exact tree
  attention; sums over all root-to-leaf paths);
* positive integer ``N`` -> ``mode="stochastic"`` with ``num_samples=N``
  (Gumbel-max path sampling).

The :mod:`attn_bench.plot_seqlen` and :mod:`attn_bench.run` CLIs expose a
``--treeattn-num-samples`` flag that sets this variable for the duration
of the run.
"""

from __future__ import annotations

from typing import Any

import torch

from attn_bench.adapters import KernelAdapter
from attn_bench.baselines._treeattn_common import (
    from_sbhd,
    haar_internal_node_keys,
    is_power_of_two,
    resolve_num_samples,
    to_sbhd,
)


BASELINE_NAME = "treeattn_torch"


def _hierarchical_attention():
    from treeattn_torch import hierarchical_attention
    return hierarchical_attention


def _call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,  # tree attention has its own implicit scaling
) -> torch.Tensor:
    hierarchical_attention = _hierarchical_attention()
    num_samples = resolve_num_samples()
    mode = "stochastic" if num_samples is not None else "deterministic"

    B, S, H, D = q.shape
    if not is_power_of_two(S):
        raise ValueError(f"treeattn_torch requires power-of-two seqlen, got {S}")

    q_sbhd = to_sbhd(q)
    k_sbhd = to_sbhd(k)
    v_sbhd = to_sbhd(v)

    # Internal-node keys: K = N-1 from Haar transform of input keys.
    k_internal = haar_internal_node_keys(k_sbhd)  # (S-1, B, H, D)

    E = H * D
    # hierarchical_attention takes (L,B,E), (K,B,E), (N,B,E) with packed heads.
    query = q_sbhd.reshape(S, B, E)
    key = k_internal.reshape(S - 1, B, E)
    value = v_sbhd.reshape(S, B, E)

    kwargs: dict[str, Any] = dict(
        query=query,
        key=key,
        value=value,
        num_heads=H,
        is_causal=causal,
        mode=mode,
    )
    if mode == "stochastic":
        kwargs["num_samples"] = num_samples

    out_lbe, _weights = hierarchical_attention(**kwargs)
    # out_lbe: (L=S, B, E). Reshape to (S, B, H, D) then to (B, S, H, D).
    out_sbhd = out_lbe.reshape(S, B, H, D)
    return from_sbhd(out_sbhd)


def adapter() -> KernelAdapter:
    return KernelAdapter(
        name=BASELINE_NAME,
        fn=_call,
        # hierarchical_attention is autograd-traceable, but backward through
        # the per-level Python loop is expensive; keep it enabled so users
        # see honest numbers.
        supports_backward=True,
        supports_causal=True,
        allowed_dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
        allowed_head_dims=frozenset({32, 64, 96, 128}),
        layout="bshd",
    )


def is_available() -> bool:
    try:
        _hierarchical_attention()
        return True
    except Exception:
        return False
