"""Result reporting: CSV + JSON writers and a pretty stdout table."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

from attn_bench.harness import Result


PRIMARY_BASELINE = "flash_attn_2"


def _speedup(baseline_ms: float | None, candidate_ms: float | None) -> float:
    if (
        baseline_ms is None
        or candidate_ms is None
        or candidate_ms <= 0
        or math.isnan(baseline_ms)
        or math.isnan(candidate_ms)
    ):
        return float("nan")
    return baseline_ms / candidate_ms


def annotate_speedups(results: list[Result], baseline_name: str = PRIMARY_BASELINE) -> list[dict]:
    """Build flat row dicts with speedup-vs-baseline columns."""
    # Index baseline by config key.
    by_cfg: dict[tuple, Result] = {}
    for r in results:
        if r.kernel == baseline_name and r.status == "ok":
            by_cfg[_cfg_key(r)] = r

    rows: list[dict] = []
    for r in results:
        row = r.row()
        base = by_cfg.get(_cfg_key(r))
        if base is not None and r.kernel != baseline_name and r.status == "ok":
            row["speedup_fwd_inference_vs_" + baseline_name] = _speedup(
                base.fwd_inference.median_ms if base.fwd_inference else None,
                r.fwd_inference.median_ms if r.fwd_inference else None,
            )
            row["speedup_step_vs_" + baseline_name] = _speedup(
                base.step.median_ms if base.step else None,
                r.step.median_ms if r.step else None,
            )
        rows.append(row)
    return rows


def _cfg_key(r: Result) -> tuple:
    c = r.config
    return (c.batch, c.seqlen, c.nheads, c.head_dim, c.dtype, c.causal)


def write_csv(rows: Iterable[dict], path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        Path(path).write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(rows: Iterable[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(rows), indent=2, default=str))


def _fmt(x, spec: str = ".3f") -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "-"
        return format(x, spec)
    return str(x)


def print_table(rows: list[dict], baseline_name: str = PRIMARY_BASELINE) -> None:
    """Print a compact stdout table; uses ``rich`` if available, else plain."""
    sp_fwd_key = f"speedup_fwd_inference_vs_{baseline_name}"
    sp_step_key = f"speedup_step_vs_{baseline_name}"

    cols: list[tuple[str, str, "callable"]] = [
        ("kernel",  "left",   lambda r: str(r.get("kernel", "-"))),
        ("dtype",   "left",   lambda r: str(r.get("dtype", "-"))),
        ("b",       "right",  lambda r: str(r.get("batch", "-"))),
        ("s",       "right",  lambda r: str(r.get("seqlen", "-"))),
        ("h",       "right",  lambda r: str(r.get("nheads", "-"))),
        ("d",       "right",  lambda r: str(r.get("head_dim", "-"))),
        ("causal",  "center", lambda r: str(r.get("causal", "-"))),
        ("status",  "left",   lambda r: str(r.get("status", "-"))),
        ("inf_ms",  "right",  lambda r: _fmt(r.get("fwd_inference_ms_median"), ".3f")),
        ("step_ms", "right",  lambda r: _fmt(r.get("step_ms_median"), ".3f")),
        ("inf_TF/s","right",  lambda r: _fmt(r.get("fwd_inference_tflops"), ".1f")),
        ("step_TF/s","right", lambda r: _fmt(r.get("step_tflops"), ".1f")),
        ("fwd_mem_MiB","right", lambda r: _fmt(r.get("fwd_peak_mem_mb"), ".1f")),
        ("saved_MiB","right", lambda r: _fmt(r.get("fwd_saved_mem_mb"), ".1f")),
        ("bwd_mem_MiB","right", lambda r: _fmt(r.get("bwd_peak_mem_mb"), ".1f")),
        (f"sp_inf/{baseline_name}", "right", lambda r: _fmt(r.get(sp_fwd_key), ".2f")),
        (f"sp_step/{baseline_name}", "right", lambda r: _fmt(r.get(sp_step_key), ".2f")),
    ]

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(show_lines=False, header_style="bold")
        for name, just, _ in cols:
            table.add_column(name, justify=just)
        for r in rows:
            table.add_row(*[accessor(r) for _, _, accessor in cols])
        console.print(table)
    except ImportError:  # pragma: no cover
        print("\t".join(name for name, _, _ in cols))
        for r in rows:
            print("\t".join(accessor(r) for _, _, accessor in cols))
