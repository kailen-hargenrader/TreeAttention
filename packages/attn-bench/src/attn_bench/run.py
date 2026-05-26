"""CLI entrypoint: ``uv run python -m attn_bench.run --kernel pkg.mod:fn``.

Runs every configured baseline (default: FlashAttention-2) and the user's
kernel-under-test across the sweep, writes CSV + JSON to ``--output-dir``,
and prints a pretty results table with speedup-vs-FA2 columns.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import torch

from attn_bench.adapters import KernelAdapter, load_entrypoint
from attn_bench.baselines import BUILTIN_BASELINES, get_baseline
from attn_bench.configs import (
    BenchConfig,
    build_sweep,
    default_sweep,
    dtype_str,
    load_sweep,
    parse_dtype,
)
from attn_bench.harness import run_sweep
from attn_bench.reporting import (
    PRIMARY_BASELINE,
    annotate_speedups,
    print_table,
    write_csv,
    write_json,
)


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_bool_list(s: str) -> list[bool]:
    out: list[bool] = []
    for x in s.split(","):
        x = x.strip().lower()
        if x in {"true", "t", "1", "yes", "y"}:
            out.append(True)
        elif x in {"false", "f", "0", "no", "n"}:
            out.append(False)
    return out


def _parse_dtype_list(s: str) -> list[torch.dtype]:
    return [parse_dtype(x) for x in s.split(",") if x.strip()]


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
        prog="attn-bench",
        description="Benchmark an attention CUDA kernel against FlashAttention-2 on A100.",
    )
    p.add_argument(
        "--kernel",
        help="Entrypoint of the kernel-under-test, e.g. 'my_pkg.mod:attn_fn' "
             "(omit to benchmark only the baselines).",
    )
    p.add_argument(
        "--baselines",
        default=PRIMARY_BASELINE,
        help=f"Comma-separated baseline names. Available: {','.join(BUILTIN_BASELINES)}. "
             f"Default: {PRIMARY_BASELINE}.",
    )
    p.add_argument("--config", help="Path to YAML/JSON sweep config (overrides defaults).")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations per cell.")
    p.add_argument("--iters", type=int, default=50, help="Measured iterations per cell.")
    p.add_argument("--output-dir", default="results", help="Directory for CSV+JSON outputs.")
    p.add_argument("--no-backward", action="store_true", help="Skip backward-pass timing.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--treeattn-num-samples",
        type=int,
        default=None,
        help="If set, run treeattn_torch / treeattn_jax in stochastic mode "
             "with this many sampled paths per query (sets "
             "TREEATTN_NUM_SAMPLES). "
             "Default: deterministic (exact).",
    )

    # Sweep overrides (apply on top of defaults; ignored if --config is set).
    p.add_argument("--batches", type=_parse_int_list, default=None, help="e.g. '1,2'")
    p.add_argument("--nheads", type=_parse_int_list, default=None, help="e.g. '8,16'")
    p.add_argument("--head-dims", type=_parse_int_list, default=None, help="e.g. '64,128'")
    p.add_argument("--seqlens", type=_parse_int_list, default=None, help="e.g. '512,1024,2048'")
    p.add_argument("--causal", type=_parse_bool_list, default=None, help="e.g. 'false,true'")
    p.add_argument("--dtypes", type=_parse_dtype_list, default=None, help="e.g. 'bf16,fp16'")

    return p


def _build_sweep_from_args(args: argparse.Namespace) -> list[BenchConfig]:
    if args.config:
        return load_sweep(args.config)
    if any(
        x is not None
        for x in (args.batches, args.nheads, args.head_dims, args.seqlens, args.causal, args.dtypes)
    ):
        return build_sweep(
            batches=args.batches or [2],
            nheads_list=args.nheads or [16],
            head_dims=args.head_dims or [64, 128],
            seqlens=args.seqlens or [512, 1024, 2048, 4096, 8192, 16384],
            causals=args.causal if args.causal is not None else [False, True],
            dtypes=args.dtypes or [torch.bfloat16],
        )
    return default_sweep()


def _collect_adapters(args: argparse.Namespace) -> list[KernelAdapter]:
    adapters: list[KernelAdapter] = []
    seen: set[str] = set()

    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]
    for name in baseline_names:
        try:
            ad = get_baseline(name)
        except KeyError as e:
            print(f"[warn] {e}", file=sys.stderr)
            continue
        adapters.append(ad)
        seen.add(ad.name)

    if args.kernel:
        ad = load_entrypoint(args.kernel)
        if ad.name in seen:
            ad = KernelAdapter(
                name=ad.name + "_user", fn=ad.fn,
                supports_backward=ad.supports_backward,
                supports_causal=ad.supports_causal,
                allowed_dtypes=ad.allowed_dtypes,
                allowed_head_dims=ad.allowed_head_dims,
                layout=ad.layout,
            )
        adapters.append(ad)

    return adapters


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
        print("ERROR: no adapters to benchmark (specify --kernel or valid --baselines).",
              file=sys.stderr)
        return 2

    sweep = _build_sweep_from_args(args)

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
    print(f"[run] sweep: {len(sweep)} configs x {len(adapters)} kernels "
          f"= {len(sweep) * len(adapters)} cells")

    def on_result(r):
        marker = {"ok": "✓", "oom": "OOM", "skipped": "skip", "error": "ERR"}.get(r.status, r.status)
        fwd = r.fwd.median_ms if r.fwd else float("nan")
        bwd = r.bwd.median_ms if r.bwd else float("nan")
        print(f"  [{marker:>4}] {r.kernel:<24} {r.config.label}  "
              f"fwd={fwd:.3f}ms  bwd={bwd:.3f}ms"
              + (f"  ({r.skip_reason or r.error_msg})" if r.status != "ok" else ""))

    results = run_sweep(
        adapters, sweep,
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
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
