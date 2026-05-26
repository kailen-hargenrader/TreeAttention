# treeattn-jax

JAX-backed hierarchical (tree) attention with a `torch.autograd.Function`
bridge (zero-copy via DLPack).

Vendored from `tachyon.fnx.tree_mha_jax` and `tachyon.modules.fnx`.
Kept: the four deterministic/stochastic × causal/non-causal entrypoints
and their custom-VJP helpers. Dropped: fused variants, Haar transform,
the original-namespace `nn.Module` wrappers.

## Public API

```python
from treeattn_jax import (
    jax_tree_attention_noncausal_deterministic,
    jax_tree_attention_causal_deterministic,
    jax_tree_attention_noncausal_stochastic,
    jax_tree_attention_causal_stochastic,
)
```

All four take `q: (L,B,H,D)`, `k: (K,B,H,D)`, `v: (N,B,H,D)`.
