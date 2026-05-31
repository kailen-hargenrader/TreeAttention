"""Built-in baseline adapters keyed by short name."""

from __future__ import annotations

from typing import Callable

from attn_bench.adapters import KernelAdapter
from attn_bench.baselines import flash_attn2, sdpa, treeattn_cuda, treeattn_jax, treeattn_torch


BUILTIN_BASELINES: dict[str, Callable[[], KernelAdapter]] = {
    flash_attn2.BASELINE_NAME: flash_attn2.adapter,
    sdpa.BASELINE_NAME: sdpa.adapter,
    treeattn_cuda.BASELINE_NAME: treeattn_cuda.adapter,
    treeattn_torch.BASELINE_NAME: treeattn_torch.adapter,
    treeattn_jax.BASELINE_NAME: treeattn_jax.adapter,
}


def get_baseline(name: str) -> KernelAdapter:
    if name not in BUILTIN_BASELINES:
        raise KeyError(
            f"unknown baseline {name!r}; known: {sorted(BUILTIN_BASELINES)}"
        )
    return BUILTIN_BASELINES[name]()


__all__ = [
    "BUILTIN_BASELINES",
    "get_baseline",
    "flash_attn2",
    "sdpa",
    "treeattn_cuda",
    "treeattn_torch",
    "treeattn_jax",
]
