"""Sweep configuration: ``BenchConfig`` dataclass + default sweep + YAML loader."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch


_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    key = name.lower().strip()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype)


@dataclass(frozen=True)
class BenchConfig:
    """A single benchmark point: shape + dtype + causal flag."""

    batch: int
    seqlen: int
    nheads: int
    head_dim: int
    dtype: torch.dtype
    causal: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dtype"] = dtype_str(self.dtype)
        return d

    @property
    def label(self) -> str:
        return (
            f"b={self.batch} s={self.seqlen} h={self.nheads} d={self.head_dim} "
            f"dtype={dtype_str(self.dtype)} causal={self.causal}"
        )


def default_sweep() -> list[BenchConfig]:
    """The default sweep used when ``--config`` is not provided.

    Matches the shape grid commonly used in FA2's own A100 benchmarks, in
    bf16 (primary) and fp16 (secondary).
    """
    batches = [1]
    nheads_list = [2]
    head_dims = [64]
    seqlens = [1 << e for e in range(11, 21)]
    causals = [False]
    dtypes = [torch.bfloat16, torch.float16]

    out: list[BenchConfig] = []
    for dtype in dtypes:
        for b in batches:
            for h in nheads_list:
                for d in head_dims:
                    for s in seqlens:
                        for c in causals:
                            out.append(
                                BenchConfig(
                                    batch=b, seqlen=s, nheads=h,
                                    head_dim=d, dtype=dtype, causal=c,
                                )
                            )
    return out


def build_sweep(
    *,
    batches: Iterable[int] = (1,),
    nheads_list: Iterable[int] = (2,),
    head_dims: Iterable[int] = (64,),
    seqlens: Iterable[int] = tuple(1 << e for e in range(11, 21)),
    causals: Iterable[bool] = (False,),
    dtypes: Iterable[torch.dtype] = (torch.bfloat16,),
) -> list[BenchConfig]:
    """Cartesian-product builder used by the CLI for ``--seqlens`` etc."""
    out: list[BenchConfig] = []
    for dtype in dtypes:
        for b in batches:
            for h in nheads_list:
                for d in head_dims:
                    for s in seqlens:
                        for c in causals:
                            out.append(
                                BenchConfig(
                                    batch=b, seqlen=s, nheads=h,
                                    head_dim=d, dtype=dtype, causal=c,
                                )
                            )
    return out


def load_sweep(path: str | Path) -> list[BenchConfig]:
    """Load a sweep from a YAML or JSON file.

    Expected schema (top-level dict)::

        batches: [2]
        nheads: [16]
        head_dims: [64, 128]
        seqlens: [512, 1024, 2048, 4096, 8192]
        causals: [false, true]
        dtypes: [bf16, fp16]

    Anything missing falls back to the corresponding default.
    Alternatively the file may be a list of explicit per-config dicts.
    """
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if isinstance(data, list):
        return [
            BenchConfig(
                batch=int(d["batch"]),
                seqlen=int(d["seqlen"]),
                nheads=int(d["nheads"]),
                head_dim=int(d["head_dim"]),
                dtype=parse_dtype(d.get("dtype", "bf16")),
                causal=bool(d.get("causal", False)),
            )
            for d in data
        ]

    return build_sweep(
        batches=data.get("batches", [1]),
        nheads_list=data.get("nheads", [2]),
        head_dims=data.get("head_dims", [64]),
        seqlens=data.get("seqlens", [1 << e for e in range(11, 21)]),
        causals=data.get("causals", [False]),
        dtypes=[parse_dtype(d) for d in data.get("dtypes", ["bf16"])],
    )
