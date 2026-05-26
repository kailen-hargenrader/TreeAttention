"""FLOP accounting for attention forward / backward.

Convention follows the FlashAttention paper and ``flash-attn`` benchmarks:

* Forward (no causal mask): ``4 * batch * nheads * seqlen^2 * head_dim``
  (2 matmuls: ``Q @ K^T`` and ``P @ V``, each ``2 * b * h * s * s * d`` FLOPs).
* Causal halves this (only the lower-triangle of the score matrix is used).
* Backward: ``2.5 *`` forward (gradients for Q, K, V plus the recomputation
  done by FA-style kernels).

These counts are theoretical / nominal — useful for TFLOP/s comparisons
between kernels with identical algorithmic complexity. They do not account
for online-softmax scratch work, head_dim-K rounding, etc.
"""

from __future__ import annotations


def fwd_flops(batch: int, seqlen: int, nheads: int, head_dim: int, causal: bool) -> float:
    flops = 4.0 * batch * nheads * seqlen * seqlen * head_dim
    if causal:
        flops *= 0.5
    return flops


def bwd_flops(batch: int, seqlen: int, nheads: int, head_dim: int, causal: bool) -> float:
    return 2.5 * fwd_flops(batch, seqlen, nheads, head_dim, causal)


def tflops_per_sec(flops: float, elapsed_ms: float) -> float:
    if elapsed_ms <= 0.0:
        return float("nan")
    return flops / (elapsed_ms * 1e-3) / 1e12
