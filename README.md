# TreeAttention

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
containing two packages:

| Package | Path | Status | Purpose |
|---|---|---|---|
| `attn-bench` | [packages/attn-bench/](packages/attn-bench/) | implemented | Benchmark harness for attention CUDA kernels on A100, with FlashAttention-2 as the baseline. |
| `treeattn` | [packages/treeattn/](packages/treeattn/) | **placeholder** | Tree-based attention model. Not yet implemented — only the package name and import path are reserved. |

The rest of this README documents the benchmark harness. Hardware target:
A100 (compute capability ≥ 8.0). Precision: `bfloat16` (default) and
`float16` — the dtypes FA2 supports. Package management: **`uv`**.

---

## Install

This project uses [`uv`](https://docs.astral.sh/uv/) for everything.

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install the harness (torch 2.7.1+cu126, pyyaml, rich, numpy)
uv sync

# 3. Install the FlashAttention-2 baseline (prebuilt wheel, ~244 MiB)
uv sync --group fa2
```

### Why these versions?

The workspace root `pyproject.toml` pins:

- **Python 3.11** (`requires-python = ">=3.11,<3.12"`, set per-package)
- **`torch==2.7.1`** from the PyTorch cu12.6 index (`tool.uv.index "pytorch-cu126"`)
- **`flash-attn==2.8.3`** from a direct GitHub-releases URL — the prebuilt
  `cu12torch2.7cxx11abiTRUE-cp311` wheel that exactly matches the torch
  above.

This combo avoids the ~30–60 min source build of `flash-attn`. The
[FlashAttention release page](https://github.com/Dao-AILab/flash-attention/releases)
ships prebuilt wheels for Python 3.9–3.13 against torch 2.4–2.9 with cu12;
if you need a different stack, find a matching wheel name there and edit
the URL in `[tool.uv.sources] flash-attn`.

A100 has compute capability 8.0, which is in the wheels' `TORCH_CUDA_ARCH_LIST`
out of the box. CUDA 12.6 binaries run fine on newer drivers (the host here
ships CUDA 13.0, which is backwards-compatible).

Verify the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
# → NVIDIA A100-SXM4-80GB
```

---

## Quickstart

Benchmark a custom kernel against FA2 over the default sweep:

```bash
uv run python -m attn_bench.run --kernel my_pkg.attn:my_attn_fn
```

Benchmark only the baselines:

```bash
uv run python -m attn_bench.run --baselines flash_attn_2,sdpa_efficient
```

Restrict the sweep:

```bash
uv run python -m attn_bench.run \
    --kernel my_pkg.attn:my_attn_fn \
    --seqlens 1024,2048,4096 \
    --head-dims 64,128 \
    --dtypes bf16 \
    --causal false,true \
    --warmup 10 --iters 50
```

Outputs (in `results/<run_id>/`):

- `results.csv` — one row per (kernel, config) cell.
- `results.json` — same data, JSON-encoded.
- `meta.json` — run metadata (device, torch version, sweep definition).

The CLI also prints a Rich table to stdout with `fwd_ms`, `bwd_ms`,
TFLOP/s, and speedup-vs-FA2 columns.

---

## Kernel contract — required for fair comparison with FA2

Your kernel **must** conform to the contract below; otherwise the numbers it
produces are not meaningfully comparable to FA2's. The harness validates
the programmatically-checkable subset and skips ineligible configs; the
rest is the kernel author's responsibility.

### Signature

```python
def attn_fn(
    q: torch.Tensor,        # (batch, seqlen, nheads, head_dim)
    k: torch.Tensor,        # same shape
    v: torch.Tensor,        # same shape
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:          # (batch, seqlen, nheads, head_dim), same dtype
    ...
```

### Required properties

| Aspect | Required behavior |
|---|---|
| Input dtype | `torch.float16` or `torch.bfloat16` (matches sweep dtype). |
| Output dtype | Same as inputs. |
| Layout | `(batch, seqlen, nheads, head_dim)` — FA2's "bshd" layout, contiguous. |
| Device | All inputs on `cuda`. Adapter must not implicitly move tensors. |
| `softmax_scale=None` | Default to `1 / sqrt(head_dim)` (FA2 convention). |
| `causal=True` | Bottom-right-aligned lower-triangular mask (FA2 convention when `seqlen_q == seqlen_k`). |
| Dropout | Always `0.0` (harness does not pass any). |
| Masking | No additive / key-padding masks in v1. |
| Heads | MHA only: `nheads_q == nheads_k == nheads_v`. (GQA/MQA out of scope.) |
| `head_dim` | One of `{32, 64, 96, 128}`. |
| In-place mods | **Forbidden.** Kernel must not modify `q`, `k`, or `v`. The harness checks `data_ptr` is unchanged after each call. |
| Backward | Implement via `torch.autograd.Function` (or any autograd-compatible mechanism) so `out.backward(grad_out)` works. If unsupported, set `supports_backward=False` on the adapter; the harness will time forward only. |
| Determinism (fwd) | Forward must be deterministic given fixed inputs. |
| No syncs in hot path | No `.item()`, no host prints, no `cudaDeviceSynchronize` inside the kernel call — the harness owns synchronization. |

### Registering the kernel

The `--kernel` flag accepts a `pkg.module:attr` entrypoint that may resolve
to either:

1. A `KernelAdapter` instance (preferred — lets you self-declare metadata).
2. A zero-arg factory returning a `KernelAdapter`.
3. A raw callable matching the signature above (wrapped in a default
   adapter; uses fp16/bf16 + head_dim {32,64,96,128} as the supported set).

Example:

```python
# my_pkg/attn.py
import torch
from attn_bench import KernelAdapter

class MyAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale):
        ...
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ...
        return dq, dk, dv, None, None

def my_attn_fn(q, k, v, causal, softmax_scale):
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    return MyAttnFn.apply(q, k, v, causal, softmax_scale)

adapter = KernelAdapter(
    name="my_attn",
    fn=my_attn_fn,
    supports_backward=True,
    supports_causal=True,
    allowed_dtypes=frozenset({torch.float16, torch.bfloat16}),
    allowed_head_dims=frozenset({32, 64, 96, 128}),
)
```

Then run:

```bash
uv run python -m attn_bench.run --kernel my_pkg.attn:adapter
```

---

## Plotting: sequence length vs latency

A companion script benchmarks **multiple kernels** over a range of sequence
lengths and produces a two-panel plot (forward / backward) plus a CSV. Both
passes are timed in a single run.

```bash
uv run python -m attn_bench.plot_seqlen \
    --kernel mine=my_pkg.attn:adapter \
    --kernel theirs=their_pkg.attn:adapter \
    --baselines flash_attn_2,sdpa_efficient \
    --seqlens 512,1024,2048,4096,8192,16384 \
    --batch 2 --nheads 16 --head-dim 128 \
    --dtype bf16 \
    --causal \
    --warmup 10 --iters 50
```

- `--kernel` is repeatable; values are either `pkg.mod:attr` (label inferred)
  or `label=pkg.mod:attr` (custom label, used in the plot legend).
- `--baselines` accepts a comma-separated list; pass `""` to disable.
- `--no-backward` skips the backward panel (forward only).
- `--linear-x` / `--linear-y` switch the corresponding axis to linear scale
  (default is log-2 on x, log-10 on y).
- Output is written to `results/seqlen-<run_id>/`:
  - `results.csv`, `results.json` — same schema as the main `run` CLI.
  - `seqlen_vs_time.png` (or the path passed to `--out`).

---

## What the harness measures

For each `(kernel, config)` cell:

1. **Inputs** are created on CUDA in the configured dtype with a fixed seed.
2. **Validation** checks dtype, shape, contiguity, device.
3. **Warmup**: `--warmup` (default 10) calls, then `cuda.synchronize()`.
4. **Forward timing**: `--iters` (default 50) calls, each bracketed by a
   `cuda.Event` pair, then a single trailing `cuda.synchronize()`.
5. **In-place check**: asserts `q/k/v.data_ptr()` are unchanged after the
   forward calls.
6. **Backward timing** (if `supports_backward`): same protocol on
   `out.backward(grad_out)`-including-forward; reported bwd time is
   `median(step) − median(fwd)`, matching the FA paper.
7. **Statistics**: median / mean / std / p10 / p90 in milliseconds; derived
   TFLOP/s using `4·b·h·s²·d` (halved for causal) for fwd and `2.5×` that
   for bwd.

TF32 and cuDNN benchmark are forced off at import time so timings are
deterministic and "fp32" (in any reference code paths) means real fp32.

OOMs and exceptions are caught per-cell and recorded as `oom` / `error`
rows; the sweep continues.

---

## Sweep configuration

Default sweep (in `attn_bench/configs.py:default_sweep`):

- `batch = 2`
- `nheads = 16`
- `head_dim ∈ {64, 128}`
- `seqlen ∈ {512, 1024, 2048, 4096, 8192, 16384}`
- `causal ∈ {False, True}`
- `dtype ∈ {bf16, fp16}`

= 192 configs.

Override via CLI flags (`--seqlens`, `--head-dims`, `--batches`, `--nheads`,
`--causal`, `--dtypes`) or pass a YAML/JSON config with `--config`:

```yaml
# sweep.yaml
batches: [2]
nheads: [16]
head_dims: [64, 128]
seqlens: [1024, 2048, 4096]
causals: [true]
dtypes: [bf16]
```

```bash
uv run python -m attn_bench.run --kernel my_pkg:adapter --config sweep.yaml
```

---

## Limitations / out of scope for v1

- No numerical correctness check vs FA2 (the harness times only).
- No memory profiling.
- No multi-GPU / distributed benchmarks.
- No GQA / MQA / sliding-window / paged / variable-length sequences.
- No dropout, no additive masks, no key-padding masks.
- `head_dim` restricted to `{32, 64, 96, 128}` — the A100-fast set.

Each of these can be added by extending `KernelAdapter` and the harness
without changing the public API.

---

## Development

```bash
uv sync --group dev
uv run pytest -v
```

The smoke test runs one tiny config end-to-end; it skips gracefully if
CUDA or `flash-attn` is unavailable.
