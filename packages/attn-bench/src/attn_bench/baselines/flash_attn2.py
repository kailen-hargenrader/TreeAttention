"""FlashAttention-2 baseline adapter.

Wraps :func:`flash_attn.flash_attn_func` (FA2). Requires ``flash-attn``
to be installed::

    uv sync --group fa2

FA2's kernels accept only fp16 / bf16 inputs in the ``(b, s, h, d)`` layout
with ``head_dim`` in ``{32, 64, 96, 128}`` on A100 (FA2 also supports 160/192/224/256
on newer hardware but those are out of scope for v1).
"""

from __future__ import annotations

import torch

from attn_bench.adapters import KernelAdapter


BASELINE_NAME = "flash_attn_2"


def _flash_attn_func_or_raise():
    try:
        from flash_attn import flash_attn_func  # type: ignore
    except Exception as e:  # pragma: no cover - exercised only without flash-attn
        raise ImportError(
            "flash-attn is not installed. Install with:\n"
            "  uv sync --group fa2\n"
            "(this compiles flash-attn from source; takes several minutes; "
            "set MAX_JOBS=4 to bound parallelism)"
        ) from e
    return flash_attn_func


def _fa2_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    flash_attn_func = _flash_attn_func_or_raise()
    return flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def adapter() -> KernelAdapter:
    """Return the FA2 KernelAdapter. Importing flash-attn is deferred until call."""
    return KernelAdapter(
        name=BASELINE_NAME,
        fn=_fa2_call,
        supports_backward=True,
        supports_causal=True,
        allowed_dtypes=frozenset({torch.float16, torch.bfloat16}),
        allowed_head_dims=frozenset({32, 64, 96, 128}),
        layout="bshd",
    )


def is_available() -> bool:
    try:
        _flash_attn_func_or_raise()
        return True
    except ImportError:
        return False
