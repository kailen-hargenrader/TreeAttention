"""Optional secondary baseline: PyTorch SDPA (memory-efficient backend).

Useful for sanity-checking FA2 numbers and providing a fallback baseline
when ``flash-attn`` is not installed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from attn_bench.adapters import KernelAdapter


BASELINE_NAME = "sdpa_efficient"


def _sdpa_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    # Repack (b, s, h, d) -> (b, h, s, d) for SDPA's API, then back.
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out = F.scaled_dot_product_attention(
                qh, kh, vh,
                is_causal=causal,
                scale=softmax_scale,
            )
    except ImportError:  # pragma: no cover - very old torch
        out = F.scaled_dot_product_attention(
            qh, kh, vh,
            is_causal=causal,
            scale=softmax_scale,
        )
    return out.transpose(1, 2).contiguous()


def adapter() -> KernelAdapter:
    return KernelAdapter(
        name=BASELINE_NAME,
        fn=_sdpa_call,
        supports_backward=True,
        supports_causal=True,
        allowed_dtypes=frozenset({torch.float16, torch.bfloat16}),
        allowed_head_dims=frozenset({32, 64, 96, 128}),
        layout="bshd",
    )
