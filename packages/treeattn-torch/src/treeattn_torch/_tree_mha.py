"""Hierarchical (tree) multi-head attention -- pure PyTorch.

Vendored from ``tachyon.modules.tree_mha`` (commit at the time of import).

Differences from upstream:

* Removed ``@jaxtyped`` / ``@typeguard.typechecked`` decorators and the
  ``jaxtyping`` / ``typeguard`` imports. The runtime shape checks they
  provided are not useful for benchmarking and interact poorly with newer
  ``typeguard`` AST rewrites.
* Removed the top-of-file imports of ``fast_hadamard_transform``,
  ``scipy``, and ``tachyon.modules.transforms.KeyHaarTransform`` -- none
  of them are used by the kept entrypoints.
* **Bug fix** in :func:`_sample_paths_non_causal` and
  :func:`_sample_paths_causal`: the upstream code did

  .. code:: python

      k_gathered = k[node_indices]               # k: (K,B,H,D), node_indices: (L*B*H*S,)
      k_d = rearrange(k_gathered,
          "(l b h s) b_ h_ d -> l b h s d", ...) # b_, h_ unused

  which is dimensionally wrong (the leading-axis fancy index carries
  through the B and H axes of ``k``, and the resulting "diagonal" slice
  is not what ``rearrange`` performs). It is also rejected by modern
  ``einops`` (axis names ``b_``/``h_`` end with underscores). The
  replacement below uses :func:`torch.gather` along the K axis with
  per-(b, h, s, d) indices, yielding the intended ``(L, B, H, S, D)``
  result.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


# Type Aliases (for documentation only -- no runtime checking)
# B: Batch, L: Query Seq Len, N: Value Seq Len, E: Embedding Dim
# K: Num Keys (N-1), H: Num Heads, D: Head Dim, S: Num Samples
Mode = Literal["deterministic", "stochastic"]


def _gather_keys_by_index(
    k: torch.Tensor,  # (K, B, H, D)
    indices: torch.Tensor,  # (..., B, H, S) with values in [0, K)
) -> torch.Tensor:
    """Gather key vectors per (b, h, sample) along the K axis.

    Returns a tensor of shape ``indices.shape + (D,)`` such that
    ``out[..., b, h, s, :] == k[indices[..., b, h, s], b, h, :]``.

    This replaces the buggy upstream pattern

    .. code:: python

        k[indices.flatten()]  # spurious extra (B, H) axes

    with a proper gather along axis 0 of ``k``.
    """
    K, B, H, D = k.shape
    # Promote k to broadcast over indices' leading dims.
    # k: (K, B, H, D) -> (B, H, K, D) for gather along dim=2.
    k_bhkd = k.permute(1, 2, 0, 3).contiguous()
    leading_shape = indices.shape[:-3]  # everything before (B, H, S)
    # Broadcast k to (*leading, B, H, K, D) without copying when possible.
    k_view = k_bhkd
    for _ in range(len(leading_shape)):
        k_view = k_view.unsqueeze(0)
    k_exp = k_view.expand(*leading_shape, B, H, K, D)
    # indices: (..., B, H, S) -> (..., B, H, S, D)
    idx = indices.clamp(0, K - 1).unsqueeze(-1).expand(*indices.shape, D)
    # Gather along the K axis (which is at position -2 of k_exp).
    return torch.gather(k_exp, dim=-2, index=idx)


def _sample_paths_non_causal(
    q: torch.Tensor,  # (L, B, H, D)
    k: torch.Tensor,  # (K, B, H, D)
    num_samples: int,
    num_leaves: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples S paths per query from the tree using the Gumbel-max trick.

    Returns ``(sampled_indices, accumulated_log_probs)`` of shape
    ``(L, B, H, S)``.
    """
    L, B, H, _ = q.shape
    K = num_leaves - 1
    log_N = (num_leaves - 1).bit_length() if num_leaves > 1 else 0
    device = q.device

    q_expanded = repeat(q, "l b h d -> l b h s d", s=num_samples)

    current_nodes = torch.zeros(L, B, H, num_samples, dtype=torch.long, device=device)
    accumulated_log_probs = torch.zeros(L, B, H, num_samples, device=device)

    for _ in range(log_N):
        valid_node_mask = current_nodes < K

        # Gather k vectors per (l, b, h, s) using current_nodes as the K-axis
        # index. The previous upstream code mis-used fancy indexing here.
        k_d = _gather_keys_by_index(k, current_nodes)  # (L, B, H, S, D)

        dots = einsum(q_expanded, k_d, "l b h s d, l b h s d -> l b h s")
        log_p_choices = torch.stack([F.logsigmoid(dots), F.logsigmoid(-dots)], dim=-1)

        gumbels = torch.empty_like(log_p_choices).exponential_().log_().neg_()
        directions = torch.argmax(log_p_choices + gumbels, dim=-1)

        chosen_log_probs = torch.gather(
            log_p_choices, -1, directions.unsqueeze(-1)
        ).squeeze(-1)
        accumulated_log_probs = accumulated_log_probs + torch.where(
            valid_node_mask, chosen_log_probs, torch.full_like(chosen_log_probs, float("-inf"))
        )

        current_nodes = 2 * current_nodes + 1 + directions

    sampled_indices = current_nodes - K
    return sampled_indices, accumulated_log_probs


def _sample_paths_causal(
    q: torch.Tensor,  # (L, B, H, D)
    k: torch.Tensor,  # (K, B, H, D)
    num_samples: int,
    num_leaves: int,
    causal_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples S paths per query, respecting the causal constraint via a two-pass algorithm."""
    L, B, H, _ = q.shape
    K = num_leaves - 1
    log_N = (num_leaves - 1).bit_length() if num_leaves > 1 else 0
    device = q.device

    all_sampled_indices = []
    all_accumulated_log_probs = []

    for t in range(L):
        q_t = q[t : t + 1]  # (1, B, H, D)
        causal_boundary_idx = t + causal_offset

        # --- Pass 1: bottom-up correction of log-masses on the boundary path ---
        boundary_path_nodes = []
        node_on_path = 0
        for d in range(log_N):
            if node_on_path >= K:
                break
            boundary_path_nodes.append(node_on_path)
            direction = (causal_boundary_idx >> (log_N - 1 - d)) & 1
            node_on_path = 2 * node_on_path + 1 + direction

        boundary_path_nodes_tensor = torch.tensor(
            boundary_path_nodes, dtype=torch.long, device=device
        )
        k_boundary = k[boundary_path_nodes_tensor]  # (path_len, B, H, D)
        # Modern einops rejects literal "1" axis names in einsum; use torch.einsum.
        dots_boundary = torch.einsum("obhd,pbhd->bhp", q_t, k_boundary)

        path_len = len(boundary_path_nodes)
        log_mass_on_path = torch.zeros(B, H, device=device)
        boundary_path_log_mass = torch.zeros(B, H, path_len, device=device)

        for d in range(path_len - 1, -1, -1):
            direction = (causal_boundary_idx >> (log_N - 1 - d)) & 1
            num_leaves_in_subtree = 1 << (log_N - 1 - d)

            log_p_left_raw = F.logsigmoid(dots_boundary[..., d])
            log_p_right_raw = F.logsigmoid(-dots_boundary[..., d])

            if direction == 0:
                log_p_on_path, log_p_off_path = log_p_left_raw, log_p_right_raw
                log_mass_child_on_path = log_mass_on_path
                log_mass_child_off_path = torch.full_like(log_mass_on_path, float("-inf"))
            else:
                log_p_on_path, log_p_off_path = log_p_right_raw, log_p_left_raw
                log_mass_child_on_path = log_mass_on_path
                log_mass_child_off_path = torch.full_like(
                    log_mass_on_path,
                    float(torch.log(torch.tensor(num_leaves_in_subtree, dtype=torch.float32))),
                )

            combined_log_mass = torch.logsumexp(
                torch.stack(
                    [
                        log_mass_child_on_path + log_p_on_path,
                        log_mass_child_off_path + log_p_off_path,
                    ]
                ),
                dim=0,
            )

            boundary_path_log_mass[..., d] = combined_log_mass
            log_mass_on_path = combined_log_mass

        # --- Pass 2: top-down sampling with corrected probabilities ---
        q_t_expanded = repeat(q_t, "1 b h d -> b h s d", s=num_samples)
        current_nodes = torch.zeros(B, H, num_samples, dtype=torch.long, device=device)
        accumulated_log_probs_t = torch.zeros(B, H, num_samples, device=device)

        for d in range(log_N):
            is_on_boundary_path = (d < path_len) and torch.all(
                current_nodes == boundary_path_nodes[d],
                dim=list(range(len(current_nodes.shape))),
            )

            # Gather k per (b, h, s). Upstream had the same dimensionality bug
            # as in the non-causal sampler.
            k_d = _gather_keys_by_index(k, current_nodes)  # (B, H, S, D)
            dots_d = einsum(q_t_expanded, k_d, "b h s d, b h s d -> b h s")
            log_p_left_d = F.logsigmoid(dots_d)
            log_p_right_d = F.logsigmoid(-dots_d)

            if is_on_boundary_path:
                direction_boundary = (causal_boundary_idx >> (log_N - 1 - d)) & 1
                log_mass_parent = boundary_path_log_mass[..., d].unsqueeze(-1)

                if d + 1 < path_len:
                    log_mass_child_on_path = boundary_path_log_mass[..., d + 1]
                else:
                    log_mass_child_on_path = 0.0

                num_leaves_in_subtree = 1 << (log_N - 1 - d)
                if direction_boundary == 0:
                    if isinstance(log_mass_child_on_path, float):
                        log_mass_left = torch.full(
                            (B, H, 1), log_mass_child_on_path, device=device
                        )
                    else:
                        log_mass_left = log_mass_child_on_path.unsqueeze(-1)
                    log_mass_right = torch.full_like(log_mass_left, float("-inf"))
                else:
                    log_mass_left = torch.full(
                        (B, H, 1),
                        float(torch.log(torch.tensor(num_leaves_in_subtree, dtype=torch.float32))),
                        device=device,
                    )
                    if isinstance(log_mass_child_on_path, float):
                        log_mass_right = torch.full(
                            (B, H, 1), log_mass_child_on_path, device=device
                        )
                    else:
                        log_mass_right = log_mass_child_on_path.unsqueeze(-1)

                log_p_left_corr = log_mass_left + log_p_left_d - log_mass_parent
                log_p_right_corr = log_mass_right + log_p_right_d - log_mass_parent
                final_log_p_left, final_log_p_right = log_p_left_corr, log_p_right_corr
            else:
                final_log_p_left, final_log_p_right = log_p_left_d, log_p_right_d

            log_p_choices = torch.stack([final_log_p_left, final_log_p_right], dim=-1)

            gumbels = torch.empty_like(log_p_choices).exponential_().log_().neg_()
            directions = torch.argmax(log_p_choices + gumbels, dim=-1)

            chosen_log_probs = torch.gather(
                log_p_choices, -1, directions.unsqueeze(-1)
            ).squeeze(-1)
            accumulated_log_probs_t = accumulated_log_probs_t + chosen_log_probs
            current_nodes = 2 * current_nodes + 1 + directions

        sampled_indices_t = current_nodes - K
        all_sampled_indices.append(sampled_indices_t)
        all_accumulated_log_probs.append(accumulated_log_probs_t)

    return torch.stack(all_sampled_indices), torch.stack(all_accumulated_log_probs)


def _deterministic_attention(
    q: torch.Tensor,  # (L, B, H, D)
    k: torch.Tensor,  # (K, B, H, D)
    v: torch.Tensor,  # (N, B, H, D)
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes exact attention by evaluating all paths in the tree."""
    L, B, H, _ = q.shape
    K = k.shape[0]
    N = v.shape[0]
    assert K == N - 1, "internal error, key value shape mismatch"
    device = q.device
    log_N = (K).bit_length() if K != 0 else 0

    log_path_probs = torch.zeros(L, B, H, 1, device=device)

    for d in range(log_N):
        level_start_idx = (1 << d) - 1
        level_end_idx = (1 << (d + 1)) - 1

        k_d = k[level_start_idx:level_end_idx]
        k_d = rearrange(k_d, "n b h d -> 1 b h n d")
        dots = einsum(q, k_d, "l b h d, one b h n d -> l b h n")
        log_p_choices = torch.stack([F.logsigmoid(dots), F.logsigmoid(-dots)], dim=-1)
        log_path_probs = rearrange(log_path_probs, "l b h p -> l b h p 1")
        if d == log_N - 1:
            num_left_over = log_p_choices.shape[3]
            left_over_probs = log_path_probs[:, :, :, :num_left_over] + log_p_choices
            left_over_probs = rearrange(left_over_probs, "l b h p two -> l b h (p two)")
            log_path_probs = rearrange(log_path_probs, "l b h p 1 -> l b h p")
            log_path_probs = torch.cat(
                [left_over_probs, log_path_probs[:, :, :, num_left_over:]], dim=-1
            )
        else:
            log_path_probs = log_path_probs + log_p_choices
            log_path_probs = rearrange(log_path_probs, "l b h p two -> l b h (p two)")

    if is_causal:
        causal_mask = torch.arange(N, device=device).view(1, N) > torch.arange(
            L, device=device
        ).view(L, 1)
        causal_mask = repeat(causal_mask, "l n -> l b h n", b=B, h=H)
        log_path_probs.masked_fill_(causal_mask, float("-inf"))
        attn_weights = torch.softmax(log_path_probs[..., :N], dim=-1)
    else:
        attn_weights = torch.exp(log_path_probs[..., :N])

    output = einsum(attn_weights.to(v.dtype), v, "l b h n, n b h d -> l b h d")
    output = rearrange(output, "l b h d -> l b (h d)")

    return output, attn_weights


def hierarchical_attention(
    query: torch.Tensor,  # (L, B, E)
    key: torch.Tensor,  # (K, B, E) with K = N-1
    value: torch.Tensor,  # (N, B, E)
    num_heads: int,
    is_causal: bool = False,
    mode: Mode = "stochastic",
    num_samples: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes hierarchical multi-head attention without padding.

    Args:
        query: Query tensor of shape (L, B, E).
        key: Key tensor for internal nodes, shape (N-1, B, E).
        value: Value tensor for leaves, shape (N, B, E).
        num_heads: Number of attention heads (H).
        is_causal: If True, applies a causal mask.
        mode: 'deterministic' for exact attention or 'stochastic' for
            sampling-based approximation.
        num_samples: Number of paths to sample per query in stochastic mode.

    Returns:
        A tuple ``(output, weights)`` with ``output`` of shape ``(L, B, E)``.
    """
    L, B, E = query.shape
    N, _, _ = value.shape
    H = num_heads
    D = E // H
    K = N - 1

    if K > 0 and key.shape[0] != K:
        raise ValueError(f"Key tensor length must be N-1 ({K}), but got {key.shape[0]}.")
    if mode == "stochastic" and num_samples is None:
        raise ValueError("num_samples must be provided for stochastic mode.")

    q = rearrange(query, "l b (h d) -> l b h d", h=H)
    k = (
        rearrange(key, "k b (h d) -> k b h d", h=H)
        if K > 0
        else torch.empty(0, B, H, D, device=query.device)
    )
    v = rearrange(value, "n b (h d) -> n b h d", h=H)

    if N == 0:
        return torch.zeros_like(query), torch.empty(L, B, H, 0, device=query.device)
    if N == 1:
        output = repeat(value, "1 b e -> l b e", l=L)
        weights = torch.ones(L, B, H, 1, device=query.device)
        return output, weights

    if mode == "deterministic":
        return _deterministic_attention(q, k, v, is_causal)

    # --- Stochastic Mode ---

    use_hybrid = is_causal and L > num_samples
    if use_hybrid:
        q_det, q_sto = q[:num_samples], q[num_samples:]
        output_det, _ = _deterministic_attention(q_det, k, v, is_causal=True)
        sampled_indices, path_log_probs = _sample_paths_causal(
            q_sto, k, num_samples, N, causal_offset=num_samples
        )
        q_proc, v_proc = q_sto, v
    else:
        if is_causal:
            sampled_indices, path_log_probs = _sample_paths_causal(q, k, num_samples, N)
        else:
            sampled_indices, path_log_probs = _sample_paths_non_causal(
                q, k, num_samples, N
            )
        q_proc, v_proc = q, v

    valid_sample_mask = (sampled_indices >= 0) & (sampled_indices < N)
    path_log_probs.masked_fill_(~valid_sample_mask, float("-inf"))

    attn_weights = torch.softmax(path_log_probs, dim=-1)

    indices_for_gather = sampled_indices.clamp(min=0, max=N - 1)
    # Same dimensionality fix as in the path samplers: gather v per
    # (l, b, h, s) along the leaf (N) axis. Upstream's rearrange-based
    # version assumed B,H,D would broadcast away, which they don't.
    gathered_values = _gather_keys_by_index(v_proc, indices_for_gather)  # (L,B,H,S,D)

    output_sto = einsum(attn_weights.to(gathered_values.dtype), gathered_values, "l b h s, l b h s d -> l b h d")
    output_sto = rearrange(output_sto, "l b h d -> l b (h d)")

    if use_hybrid:
        output = torch.cat([output_det, output_sto], dim=0)
        return output, attn_weights
    return output_sto, attn_weights
