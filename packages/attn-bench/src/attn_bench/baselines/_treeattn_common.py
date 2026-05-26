"""Shared helpers for treeattn (PyTorch + JAX) tree-attention baselines.

The kernels themselves now live in the first-party workspace packages
``treeattn-torch`` and ``treeattn-jax``; this module only provides the
shared (B,S,H,D) <-> (S,B,H,D) layout helpers, the autograd-safe Haar
transform used to build internal-node keys, and the env-driven
deterministic/stochastic mode toggle.

Both baselines wrap the same kernel contract used by the harness::

    out = fn(q, k, v, causal: bool, softmax_scale: float | None)

with ``q/k/v`` of shape ``(B, S, H, D)``. Tree attention internally treats
the input ``v`` as the ``N`` leaves of a balanced binary tree and the
input ``k`` as the source for the ``K = N-1`` internal-node keys, which we
materialize via a sequence-axis Haar transform. ``S`` must therefore be a
power of two.
"""

from __future__ import annotations

import os

import torch


NUM_SAMPLES_ENV_VAR = "TREEATTN_NUM_SAMPLES"


def resolve_num_samples() -> int | None:
    """Return ``num_samples`` from env, or ``None`` for deterministic mode.

    Raises ``ValueError`` if the env var is set to a non-integer or a
    negative value.
    """
    raw = os.environ.get(NUM_SAMPLES_ENV_VAR, "").strip()
    if not raw or raw == "0":
        return None
    try:
        n = int(raw)
    except ValueError as e:
        raise ValueError(
            f"{NUM_SAMPLES_ENV_VAR}={raw!r} is not an integer"
        ) from e
    if n < 0:
        raise ValueError(f"{NUM_SAMPLES_ENV_VAR}={n} must be >= 0")
    return n or None


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def haar_internal_node_keys(k_sbhd: torch.Tensor) -> torch.Tensor:
    """Compute internal-node tree keys from leaf keys.

    Args:
        k_sbhd: leaf keys, shape ``(S, B, H, D)`` with ``S`` a power of two.

    Returns:
        Internal-node keys of shape ``(S-1, B, H, D)`` -- a pure-PyTorch,
        autograd-safe Haar transform along the sequence dimension, with the
        global-average component (index 0) dropped.
    """
    S, B, H, D = k_sbhd.shape
    if not is_power_of_two(S):
        raise ValueError(
            f"treeattn requires power-of-two seqlen, got {S}"
        )
    out = k_sbhd
    levels = S.bit_length() - 1
    for level in range(levels):
        step = 1 << level
        half = S >> (level + 1)
        base = torch.arange(half, device=out.device) * (2 * step)
        idx1 = base
        idx2 = base + step
        a = out.index_select(0, idx1)
        b = out.index_select(0, idx2)
        avg = (a + b) / 2
        diff = (a - b) / 2
        new_out = out.clone()
        new_out[idx1] = avg
        new_out[idx2] = diff
        out = new_out
    return out[1:].contiguous()


def to_sbhd(t_bshd: torch.Tensor) -> torch.Tensor:
    """``(B, S, H, D) -> (S, B, H, D)`` contiguous."""
    return t_bshd.transpose(0, 1).contiguous()


def from_sbhd(t_sbhd: torch.Tensor) -> torch.Tensor:
    """``(S, B, H, D) -> (B, S, H, D)`` contiguous."""
    return t_sbhd.transpose(0, 1).contiguous()
