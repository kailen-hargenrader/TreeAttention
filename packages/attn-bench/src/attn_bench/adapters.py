"""Kernel adapter protocol + entrypoint loading + input validation.

The harness benchmarks *kernels* (callables) wrapped in :class:`KernelAdapter`.
The adapter carries enough metadata (allowed dtypes, head_dims, layout,
backward support) for the harness to validate inputs and skip ineligible
configs without crashing.

The canonical kernel signature is::

    out = fn(q, k, v, causal: bool, softmax_scale: float | None)

where ``q``, ``k``, ``v`` have shape ``(batch, seqlen, nheads, head_dim)``
and dtype is one of ``allowed_dtypes`` (typically fp16 / bf16, to match
FlashAttention-2).

See README's "Kernel contract" section for the full specification that
kernels-under-test must conform to in order to be fairly comparable to FA2.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch


KernelFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool, float | None], torch.Tensor]


@dataclass(frozen=True)
class KernelAdapter:
    """Metadata + callable for an attention kernel under benchmark."""

    name: str
    fn: KernelFn
    supports_backward: bool = True
    supports_causal: bool = True
    allowed_dtypes: frozenset[torch.dtype] = field(
        default_factory=lambda: frozenset({torch.float16, torch.bfloat16})
    )
    allowed_head_dims: frozenset[int] = field(
        default_factory=lambda: frozenset({32, 64, 96, 128})
    )
    layout: str = "bshd"

    def is_compatible(self, dtype: torch.dtype, head_dim: int, causal: bool) -> tuple[bool, str]:
        """Return (compatible, reason). reason is empty if compatible."""
        if dtype not in self.allowed_dtypes:
            allowed = ", ".join(str(d) for d in sorted(self.allowed_dtypes, key=str))
            return False, f"dtype {dtype} not in allowed_dtypes ({allowed})"
        if head_dim not in self.allowed_head_dims:
            allowed = ", ".join(str(h) for h in sorted(self.allowed_head_dims))
            return False, f"head_dim {head_dim} not in allowed_head_dims ({allowed})"
        if causal and not self.supports_causal:
            return False, "kernel does not support causal=True"
        return True, ""


def load_entrypoint(spec: str) -> KernelAdapter:
    """Load a :class:`KernelAdapter` from a ``pkg.module:attr`` entrypoint.

    The referenced attribute may be:

    * a :class:`KernelAdapter` instance, returned as-is;
    * a zero-arg factory ``() -> KernelAdapter``;
    * a raw kernel callable matching :data:`KernelFn`, which will be wrapped in
      a default ``KernelAdapter`` (name derived from the entrypoint).
    """
    if ":" not in spec:
        raise ValueError(
            f"entrypoint {spec!r} must be of the form 'pkg.module:attr'"
        )
    mod_name, attr = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    obj = getattr(module, attr)

    if isinstance(obj, KernelAdapter):
        return obj
    if callable(obj):
        try:
            maybe = obj()
        except TypeError:
            maybe = None
        if isinstance(maybe, KernelAdapter):
            return maybe
        return KernelAdapter(name=f"{mod_name}:{attr}", fn=obj)

    raise TypeError(
        f"entrypoint {spec!r} resolved to {type(obj).__name__}, expected "
        "KernelAdapter, factory, or callable"
    )


def validate_inputs(
    adapter: KernelAdapter,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Assert q/k/v conform to the kernel contract.

    Raises a clear ``ValueError`` on mismatch so the harness can record it as
    a per-config error rather than crashing the whole sweep.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} is not a torch.Tensor")
        if not t.is_cuda:
            raise ValueError(f"{name} must be on CUDA, got {t.device}")
        if t.dtype != dtype:
            raise ValueError(f"{name}.dtype={t.dtype}, expected {dtype}")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4-D (b,s,h,d), got shape {tuple(t.shape)}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q/k/v must share shape (MHA only in v1); got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )

    head_dim = q.shape[-1]
    if head_dim not in adapter.allowed_head_dims:
        allowed = sorted(adapter.allowed_head_dims)
        raise ValueError(
            f"head_dim={head_dim} not supported by adapter '{adapter.name}' "
            f"(allowed: {allowed})"
        )


def _ptr_snapshot(*tensors: torch.Tensor) -> tuple[int, ...]:
    return tuple(t.data_ptr() for t in tensors)


def check_no_inplace(
    pre: tuple[int, ...],
    tensors: Iterable[torch.Tensor],
) -> None:
    """Verify that the kernel did not swap out the storage of q/k/v.

    This is a weak check (it would not catch in-place writes that preserve
    ``data_ptr``), but it catches the common bug of returning a kernel that
    overwrites inputs by reusing their storage.
    """
    post = tuple(t.data_ptr() for t in tensors)
    if pre != post:
        raise ValueError(
            "kernel appears to have replaced storage of q/k/v "
            "(data_ptr changed); kernels must not modify inputs"
        )
