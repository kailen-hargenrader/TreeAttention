"""Sequence-length vs latency plotting script.

For each kernel (baselines + any number of user-supplied entrypoints) sweep
a list of sequence lengths at fixed (batch, heads, head_dim, dtype, causal)
and produce:

* a CSV of raw timings with the same schema as ``attn_bench.run``;
* a two-panel matplotlib figure (forward / backward) plotting seqlen vs
  median ms on log-log axes, one line per kernel.

Both forward and backward are timed in a single run (the harness does this
by default) so the two panels are filled from one pass over the sweep.

Usage::

    uv run python -m attn_bench.plot_seqlen \\
        --kernel my_pkg.attn:adapter \\
        --kernel another=other_pkg:fn \\
        --baselines flash_attn_2,sdpa_efficient \\
        --seqlens 512,1024,2048,4096,8192,16384 \\
        --batch 2 --nheads 16 --head-dim 128 \\
        --dtype bf16 --causal \\
        --out plots/seqlen_sweep.png

``--kernel`` may be passed multiple times. Each value is either
``entrypoint`` (label inferred from the entrypoint) or ``label=entrypoint``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch

from attn_bench.adapters import KernelAdapter, load_entrypoint
from attn_bench.baselines import BUILTIN_BASELINES, get_baseline
from attn_bench.configs import BenchConfig, dtype_str, parse_dtype
from attn_bench.harness import Result, run_sweep
from attn_bench.reporting import annotate_speedups, write_csv, write_json


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_kernel_spec(spec: str) -> tuple[str | None, str]:
    """Parse ``label=entrypoint`` or bare ``entrypoint``."""
    if "=" in spec and ":" in spec.split("=", 1)[1]:
        label, ep = spec.split("=", 1)
        return label.strip() or None, ep.strip()
    return None, spec.strip()


def _git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="attn-bench-plot",
        description="Plot sequence length vs forward/backward time for one or more attention kernels.",
    )
    p.add_argument(
        "--kernel", "-k", action="append", default=[],
        help="Kernel under test, repeatable. Either 'pkg.mod:attr' or 'label=pkg.mod:attr'.",
    )
    p.add_argument(
        "--baselines",
        default="flash_attn_2",
        help=f"Comma-separated baseline names. Available: {','.join(BUILTIN_BASELINES)}. "
             f"Empty string to disable. Default: flash_attn_2.",
    )
    p.add_argument(
        "--seqlens", type=_parse_int_list,
        default=[512, 1024, 2048, 4096, 8192, 16384],
        help="Comma-separated sequence lengths.",
    )
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--nheads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--causal", action="store_true", help="Use a causal mask.")
    p.add_argument("--no-backward", action="store_true", help="Skip backward (only fwd panel).")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--treeattn-num-samples",
        type=int,
        default=None,
        help="If set, run treeattn_torch / treeattn_jax in stochastic mode "
             "with this many sampled paths per query (sets "
             "TREEATTN_NUM_SAMPLES). Default: deterministic (exact).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: plots/seqlen-<run_id>.png",
    )
    p.add_argument(
        "--output-dir", default="results",
        help="Directory for CSV+JSON outputs alongside the plot.",
    )
    p.add_argument(
        "--linear-x", action="store_true",
        help="Use a linear x-axis instead of log-2.",
    )
    p.add_argument(
        "--linear-y", action="store_true",
        help="Use a linear y-axis instead of log-10.",
    )
    p.add_argument(
        "--title", default=None, help="Custom figure title.",
    )
    return p


def _collect_adapters(args: argparse.Namespace) -> list[tuple[str, KernelAdapter]]:
    """Returns list of (display_label, adapter). Display labels are unique."""
    labeled: list[tuple[str, KernelAdapter]] = []
    used: set[str] = set()

    def add(label: str, ad: KernelAdapter) -> None:
        base = label
        i = 2
        while label in used:
            label = f"{base}#{i}"
            i += 1
        used.add(label)
        labeled.append((label, ad))

    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]
    for name in baseline_names:
        try:
            add(name, get_baseline(name))
        except KeyError as e:
            print(f"[warn] {e}", file=sys.stderr)

    for spec in args.kernel:
        label, ep = _parse_kernel_spec(spec)
        ad = load_entrypoint(ep)
        add(label or ad.name, ad)

    return labeled


def _make_sweep(args: argparse.Namespace) -> list[BenchConfig]:
    dtype = parse_dtype(args.dtype)
    return [
        BenchConfig(
            batch=args.batch,
            seqlen=s,
            nheads=args.nheads,
            head_dim=args.head_dim,
            dtype=dtype,
            causal=args.causal,
        )
        for s in sorted(set(args.seqlens))
    ]


def _series_from_results(
    results: list[Result], label: str, adapter_name: str,
) -> tuple[list[int], list[float], list[float]]:
    """Return (seqlens, fwd_median_ms, bwd_median_ms) for one kernel,
    keeping only ``status=="ok"`` cells. Missing bwd is NaN.
    """
    seqs: list[int] = []
    fwds: list[float] = []
    bwds: list[float] = []
    for r in results:
        if r.kernel != adapter_name or r.status != "ok":
            continue
        seqs.append(r.config.seqlen)
        fwds.append(r.fwd.median_ms if r.fwd else float("nan"))
        bwds.append(r.bwd.median_ms if r.bwd else float("nan"))
    order = sorted(range(len(seqs)), key=lambda i: seqs[i])
    return [seqs[i] for i in order], [fwds[i] for i in order], [bwds[i] for i in order]


def _plot(
    labeled: list[tuple[str, KernelAdapter]],
    results: list[Result],
    cfg_template: BenchConfig,
    out_path: Path,
    *,
    do_backward: bool,
    log_x: bool,
    log_y: bool,
    title: str | None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if do_backward else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    ax_fwd = axes[0, 0]
    ax_bwd = axes[0, 1] if do_backward else None

    cmap = plt.get_cmap("tab10")
    for i, (label, adapter) in enumerate(labeled):
        color = cmap(i % 10)
        seqs, fwds, bwds = _series_from_results(results, label, adapter.name)
        if not seqs:
            continue
        ax_fwd.plot(seqs, fwds, marker="o", color=color, label=label)
        if ax_bwd is not None:
            ax_bwd.plot(seqs, bwds, marker="s", color=color, label=label, linestyle="--")

    for ax, panel_title in (
        (ax_fwd, "Forward"),
        *([(ax_bwd, "Backward")] if ax_bwd is not None else []),
    ):
        if log_x:
            ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Time (ms, median)")
        ax.set_title(panel_title)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    base_title = title or (
        f"Attention latency vs seqlen  "
        f"(b={cfg_template.batch}, h={cfg_template.nheads}, "
        f"d={cfg_template.head_dim}, dtype={dtype_str(cfg_template.dtype)}, "
        f"causal={cfg_template.causal})"
    )
    fig.suptitle(base_title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.treeattn_num_samples is not None:
        import os
        os.environ["TREEATTN_NUM_SAMPLES"] = str(args.treeattn_num_samples)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; this harness requires an NVIDIA GPU.", file=sys.stderr)
        return 2

    labeled = _collect_adapters(args)
    if not labeled:
        print("ERROR: no kernels to benchmark. Pass --kernel and/or --baselines.", file=sys.stderr)
        return 2

    # The harness keys results by adapter.name, so rename adapters to their
    # display label to keep series separate even when two adapters share an
    # internal name.
    relabeled: list[KernelAdapter] = []
    for label, ad in labeled:
        if ad.name == label:
            relabeled.append(ad)
        else:
            relabeled.append(KernelAdapter(
                name=label, fn=ad.fn,
                supports_backward=ad.supports_backward,
                supports_causal=ad.supports_causal,
                allowed_dtypes=ad.allowed_dtypes,
                allowed_head_dims=ad.allowed_head_dims,
                layout=ad.layout,
            ))

    sweep = _make_sweep(args)
    cfg_template = sweep[0]

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"seqlen-{ts}-{_git_sha(Path.cwd())}"
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {run_id} on {torch.cuda.get_device_name(0)}")
    print(f"[run] kernels: {[label for label, _ in labeled]}")
    print(f"[run] seqlens: {[c.seqlen for c in sweep]}  "
          f"b={args.batch} h={args.nheads} d={args.head_dim} "
          f"dtype={args.dtype} causal={args.causal}")

    def on_result(r: Result) -> None:
        marker = {"ok": "✓", "oom": "OOM", "skipped": "skip", "error": "ERR"}.get(r.status, r.status)
        fwd = r.fwd.median_ms if r.fwd else float("nan")
        bwd = r.bwd.median_ms if r.bwd else float("nan")
        print(f"  [{marker:>4}] {r.kernel:<24} s={r.config.seqlen:<6}  "
              f"fwd={fwd:.3f}ms  bwd={bwd:.3f}ms"
              + (f"  ({r.skip_reason or r.error_msg})" if r.status != "ok" else ""))

    results = run_sweep(
        relabeled, sweep,
        warmup=args.warmup, iters=args.iters,
        do_backward=not args.no_backward, seed=args.seed,
        on_result=on_result,
    )

    rows = annotate_speedups(results, baseline_name="flash_attn_2")
    write_csv(rows, out_dir / "results.csv")
    write_json(rows, out_dir / "results.json")

    out_path = Path(args.out) if args.out else (out_dir / "seqlen_vs_time.png")
    _plot(
        [(label, ad) for label, ad in zip([a.name for a in relabeled], relabeled)],
        results,
        cfg_template,
        out_path,
        do_backward=not args.no_backward,
        log_x=not args.linear_x,
        log_y=not args.linear_y,
        title=args.title,
    )

    print()
    print(f"[run] wrote {out_dir / 'results.csv'}")
    print(f"[run] wrote {out_dir / 'results.json'}")
    print(f"[run] wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
