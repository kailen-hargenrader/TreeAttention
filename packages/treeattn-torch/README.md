# treeattn-torch

Pure-PyTorch implementation of hierarchical (tree) attention.

Vendored from `tachyon.modules.tree_mha` (the original reference impl).
Differences:

1. Dropped `jaxtyping` / `typeguard` decorators and unused
   `fast_hadamard_transform` / `KeyHaarTransform` imports.
2. Fixed a dimensionality bug in `_sample_paths_{causal,non_causal}`:
   replaced fancy indexing `k[node_indices]` (which kept spurious B,H
   axes from `k: (K,B,H,D)`) with `torch.gather` along the K axis.

## Public API

```python
from treeattn_torch import hierarchical_attention

out, weights = hierarchical_attention(
    query,       # (L, B, E)
    key,         # (K, B, E)  K = N-1
    value,       # (N, B, E)  N a power of two
    num_heads=H,
    is_causal=False,
    mode="deterministic",  # or "stochastic"
    num_samples=64,
)
```
