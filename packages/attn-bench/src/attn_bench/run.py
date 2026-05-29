"""CLI entrypoint: ``uv run attn-bench --all --seqlens 2048,4096,...``

Sweep is always over sequence length; every other shape parameter is a
scalar. Adapters are selected with one boolean flag per built-in
(``--flash-attn-2``, ``--sdpa-efficient``, etc.) or ``--all``.

Outer-to-inner loop order: adapter -> seqlen -> repeats (the per-repeat
loop lives inside ``time_kernel``). When an adapter OOMs at some seqlen
the remaining larger seqlens are skipped for that adapter only.
"""

from __future__ import annotations

import os as _os

# Disable JAX's default GPU preallocation (~75% of device memory) so each
# adapter's true peak memory is observable independently. Must be set
# before `jax` is imported anywhere in the process.
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import torch

from attn_bench.adapters import KernelAdapter
from attn_bench.baselines import BUILTIN_BASELINES, get_baseline
from attn_bench.configs import BenchConfig, dtype_str, parse_dtype
from attn_bench.harness import run_sweep, run_sweep_isolated
from attn_bench.plot_seqlen import _plot as _plot_seqlen
from attn_bench.reporting import (
    PRIMARY_BASELINE,
    annotate_speedups,
    print_table,
    write_csv,
    write_json,
)


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def _flag_name(baseline: str) -> str:
    """``treeattn_torch`` -> ``--treeattn-torch``."""
    return "--" + baseline.replace("_", "-")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="attn-bench",
        description="Benchmark attention kernels across a sequence-length sweep.",
    )

    # One boolean flag per built-in adapter; flags auto-generated.
    adapters_group = p.add_argument_group("adapter selection")
    for name in BUILTIN_BASELINES:
        adapters_group.add_argument(
            _flag_name(name), dest=name, action="store_true",
            help=f"Include the {name!r} adapter.",
        )
    adapters_group.add_argument(
        "--all", dest="all_adapters", action="store_true",
        help="Run every built-in adapter.",
    )

    # Sweep axis (the only multi-valued arg).
    p.add_argument(
        "--seqlens", type=_parse_int_list,
        default=[1 << e for e in range(11, 21)],
        help="Comma-separated sequence lengths to sweep. Default: 2^11..2^20.",
    )

    # Fixed shape parameters.
    p.add_argument("--batch", type=int, default=1, help="Batch size (default: 1).")
    p.add_argument("--nheads", type=int, default=2, help="Number of attention heads (default: 2).")
    p.add_argument("--head-dim", type=int, default=64, help="Per-head dimension (default: 64).")
    p.add_argument("--dtype", default="bf16", help="bf16 | fp16 | fp32 (default: bf16).")
    causal_group = p.add_mutually_exclusive_group()
    causal_group.add_argument("--causal", dest="causal", action="store_true",
                              help="Use a causal mask.")
    causal_group.add_argument("--no-causal", dest="causal", action="store_false",
                              help="Disable causal mask (default).")
    p.set_defaults(causal=False)

    # Timing knobs.
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup iterations per (adapter, seqlen) cell.")
    p.add_argument("--iters", type=int, default=50,
                   help="Measured repeats per (adapter, seqlen) cell.")
    p.add_argument("--seed", type=int, default=0)

    # Misc.
    p.add_argument("--output-dir", default="results", help="Directory for CSV+JSON outputs.")
    p.add_argument("--no-backward", action="store_true", help="Skip backward-pass timing.")
    p.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation.")
    p.add_argument(
        "--no-isolate", dest="isolate", action="store_false",
        help="Run all adapters in-process (debug). Default isolates each adapter "
             "in its own subprocess so CUDA / framework allocators are torn down between adapters.",
    )
    p.set_defaults(isolate=True)
    p.add_argument(
        "--treeattn-num-samples", type=int, default=None,
        help="If set, run treeattn_torch / treeattn_jax in stochastic mode with this many "
             "sampled paths per query (sets TREEATTN_NUM_SAMPLES). Default: deterministic.",
    )

    return p


def _collect_adapters(args: argparse.Namespace) -> list[KernelAdapter]:
    if args.all_adapters:
        names = list(BUILTIN_BASELINES)
    else:
        names = [name for name in BUILTIN_BASELINES if getattr(args, name, False)]
    return [get_baseline(n) for n in names]


def _build_sweep(args: argparse.Namespace) -> list[BenchConfig]:
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


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.treeattn_num_samples is not None:
        import os
        os.environ["TREEATTN_NUM_SAMPLES"] = str(args.treeattn_num_samples)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; this harness requires an NVIDIA GPU.", file=sys.stderr)
        return 2

    adapters = _collect_adapters(args)
    if not adapters:
        flags = ", ".join(_flag_name(n) for n in BUILTIN_BASELINES)
        print(f"ERROR: no adapters selected. Pass --all or one of: {flags}.", file=sys.stderr)
        return 2

    sweep = _build_sweep(args)

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}-{_git_sha(Path.cwd())}"
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0)
    meta = {
        "run_id": run_id,
        "device": device_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "iters": args.iters,
        "do_backward": not args.no_backward,
        "adapters": [
            {
                "name": a.name,
                "allowed_dtypes": [dtype_str(d) for d in a.allowed_dtypes],
                "allowed_head_dims": sorted(a.allowed_head_dims),
                "supports_backward": a.supports_backward,
                "supports_causal": a.supports_causal,
            }
            for a in adapters
        ],
        "sweep": [c.to_dict() for c in sweep],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[run] {run_id} on {device_name}")
    print(f"[run] adapters: {[a.name for a in adapters]}")
    print(f"[run] sweep: {len(sweep)} seqlens x {len(adapters)} adapters "
          f"= {len(sweep) * len(adapters)} cells")

    def on_result(r):
        marker = {"ok": "ok", "oom": "OOM", "skipped": "skip", "error": "ERR"}.get(r.status, r.status)
        cfg = r.config
        shape = f"s={cfg.seqlen:<7}"
        if r.status == "ok":
            fwd = r.fwd.median_ms if r.fwd else float("nan")
            bwd = r.bwd.median_ms if r.bwd else float("nan")
            print(
                f"  [{marker:>4}] {r.kernel:<16} {shape}  "
                f"fwd={fwd:7.3f}ms  bwd={bwd:7.3f}ms  "
                f"mem(fwd/bwd)={r.fwd_peak_mem_mb:7.1f}/{r.bwd_peak_mem_mb:7.1f} MiB"
            )
        else:
            # OOM / skip / error: just the status, no details.
            print(f"  [{marker:>4}] {r.kernel:<16} {shape}")

    results = run_sweep(
        adapters, sweep,
        warmup=args.warmup, iters=args.iters,
        do_backward=not args.no_backward, seed=args.seed,
        on_result=on_result,
    ) if not args.isolate else run_sweep_isolated(
        [a.name for a in adapters], sweep,
        warmup=args.warmup, iters=args.iters,
        do_backward=not args.no_backward, seed=args.seed,
        on_result=on_result,
    )

    rows = annotate_speedups(results, baseline_name=PRIMARY_BASELINE)
    write_csv(rows, out_dir / "results.csv")
    write_json(rows, out_dir / "results.json")

    print()
    print_table(rows, baseline_name=PRIMARY_BASELINE)
    print()
    print(f"[run] wrote {out_dir / 'results.csv'}")
    print(f"[run] wrote {out_dir / 'results.json'}")

    if not args.no_plot:
        _autoplot(
            adapters, results, out_dir,
            do_backward=not args.no_backward,
            treeattn_num_samples=args.treeattn_num_samples,
        )
    return 0


def _autoplot(
    adapters, results, out_dir: Path, *, do_backward: bool,
    treeattn_num_samples: int | None = None,
) -> None:
    """Emit one PNG covering the seqlen sweep, one line per adapter."""
    ok_seqs = sorted({r.config.seqlen for r in results if r.status == "ok"})
    if len(ok_seqs) < 2:
        return
    template = results[0].config
    fname = (
        f"seqlen_b{template.batch}_h{template.nheads}_d{template.head_dim}_"
        f"{dtype_str(template.dtype)}_"
        f"{'causal' if template.causal else 'noncausal'}.png"
    )
    out_path = out_dir / fname
    suffix = f" (samples={treeattn_num_samples})" if treeattn_num_samples is not None else ""
    labeled = [
        (a.name + suffix if (suffix and "treeattn" in a.name) else a.name, a)
        for a in adapters
    ]
    try:
        _plot_seqlen(
            labeled, results, template, out_path,
            do_backward=do_backward, log_x=True, log_y=True, title=None,
        )
        print(f"[run] wrote {out_path}")
    except Exception as e:
        print(f"[warn] plot failed for {fname}: {e}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
