from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from treeattn_cuda import _native
from treeattn_torch import hierarchical_attention as reference_hierarchical_attention


Mode = Literal["deterministic", "stochastic"]
_DEFAULT_BLOCK_SIZE = 256
_DEFAULT_MAX_LOGIT = 1e3
_RUNTIME_STATS_TEMPLATE = {
    "sample_native_calls": 0,
    "sample_python_calls": 0,
    "replay_native_calls": 0,
    "replay_python_calls": 0,
    "weighted_value_native_calls": 0,
    "weighted_value_python_calls": 0,
    "scatter_grad_v_native_calls": 0,
    "scatter_grad_v_python_calls": 0,
    "grad_logit_native_calls": 0,
    "grad_logit_python_calls": 0,
    "accumulate_qk_native_calls": 0,
    "accumulate_qk_python_calls": 0,
    "accumulate_qk_all_depths_native_calls": 0,
    "accumulate_qk_all_depths_python_calls": 0,
}
_RUNTIME_STATS = dict(_RUNTIME_STATS_TEMPLATE)


def reset_runtime_stats() -> None:
    _RUNTIME_STATS.clear()
    _RUNTIME_STATS.update(_RUNTIME_STATS_TEMPLATE)


def get_runtime_stats() -> dict[str, int]:
    return dict(_RUNTIME_STATS)


def _note_runtime_usage(op: str, *, native: bool) -> None:
    suffix = "native_calls" if native else "python_calls"
    _RUNTIME_STATS[f"{op}_{suffix}"] += 1


def has_native_kernels() -> bool:
    """Return whether compiled CUDA kernels are available.

    This staged implementation starts with a Python autograd fallback and keeps
    the capability check explicit so a compiled extension can replace the
    internals later without changing the public surface.
    """
    return _native.has_native_kernels()


def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _path_bit_bytes(num_leaves: int) -> int:
    log_n = max(num_leaves.bit_length() - 1, 0)
    return (log_n + 7) // 8


def _read_path_bit(packed_paths: torch.Tensor, depth: int) -> torch.Tensor:
    byte_idx = depth // 8
    bit_offset = depth % 8
    return ((packed_paths[..., byte_idx] >> bit_offset) & 1).to(torch.int32)


def _empty_packed_paths(
    shape_prefix: tuple[int, ...],
    *,
    num_samples: int,
    num_leaves: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        (*shape_prefix, num_samples, _path_bit_bytes(num_leaves)),
        device=device,
        dtype=torch.uint8,
    )


def _gather_tree_values(tree: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather vectors from a tree axis while preserving batch/head lanes.

    Args:
        tree: ``(T, B, H, D)``
        indices: ``(..., B, H, S)`` with values in ``[0, T)``
    Returns:
        ``(..., B, H, S, D)``
    """
    t_len, batch, nheads, width = tree.shape
    tree_bhtd = tree.permute(1, 2, 0, 3).contiguous()
    leading_shape = indices.shape[:-3]

    tree_view = tree_bhtd
    for _ in leading_shape:
        tree_view = tree_view.unsqueeze(0)
    tree_expanded = tree_view.expand(*leading_shape, batch, nheads, t_len, width)

    safe_indices = indices.to(torch.int64).clamp(0, t_len - 1)
    gather_index = safe_indices.unsqueeze(-1).expand(*safe_indices.shape, width)
    return torch.gather(tree_expanded, dim=-2, index=gather_index)


def _scatter_tree_updates(num_nodes: int, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Scatter-add ``updates`` into the leading tree dimension.

    Args:
        indices: ``(L, B, H, S)``
        updates: ``(L, B, H, S, D)``
    Returns:
        ``(num_nodes, B, H, D)``
    """
    _, batch, nheads, _, width = updates.shape
    out = updates.new_zeros((batch, nheads, num_nodes, width))
    flat_indices = (
        indices.to(torch.int64)
        .permute(1, 2, 0, 3)
        .contiguous()
        .reshape(batch, nheads, -1, 1)
    )
    flat_indices = flat_indices.expand(batch, nheads, flat_indices.shape[2], width)
    flat_updates = (
        updates.permute(1, 2, 0, 3, 4).contiguous().reshape(batch, nheads, -1, width)
    )
    out.scatter_add_(2, flat_indices, flat_updates)
    return out.permute(2, 0, 1, 3).contiguous()


def _sample_paths_non_causal_streaming_python(
    q: torch.Tensor,
    k: torch.Tensor,
    num_samples: int,
    *,
    block_size: int,
    max_logit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample leaf paths without materializing ``(L, B, H, S, D)`` for all queries."""
    l_q, batch, nheads, _ = q.shape
    k_len = k.shape[0]
    num_leaves = k_len + 1
    if k_len == 0:
        sampled = torch.zeros(
            (l_q, batch, nheads, num_samples), device=q.device, dtype=torch.int32
        )
        log_probs = torch.zeros(
            (l_q, batch, nheads, num_samples),
            device=q.device,
            dtype=_accum_dtype(q.dtype),
        )
        packed_paths = _empty_packed_paths(
            (l_q, batch, nheads),
            num_samples=num_samples,
            num_leaves=num_leaves,
            device=q.device,
        )
        return sampled, log_probs, packed_paths

    accum_dtype = _accum_dtype(q.dtype)
    log_n = num_leaves.bit_length() - 1
    path_bytes = _path_bit_bytes(num_leaves)
    current_nodes = torch.zeros(
        (l_q, batch, nheads, num_samples), device=q.device, dtype=torch.int32
    )
    accumulated_log_probs = torch.zeros(
        (l_q, batch, nheads, num_samples), device=q.device, dtype=accum_dtype
    )
    packed_paths = torch.zeros(
        (l_q, batch, nheads, num_samples, path_bytes),
        device=q.device,
        dtype=torch.uint8,
    )
    q_accum = q.to(accum_dtype)
    k_accum = k.to(accum_dtype)

    for depth in range(log_n):
        gumbels = torch.empty(
            (l_q, batch, nheads, num_samples, 2),
            device=q.device,
            dtype=accum_dtype,
        )
        gumbels.exponential_()
        gumbels.log_()
        gumbels.neg_()
        for start in range(0, l_q, block_size):
            stop = min(start + block_size, l_q)
            node_block = current_nodes[start:stop]
            q_block = q_accum[start:stop]
            k_block = _gather_tree_values(k_accum, node_block)
            logits = (q_block.unsqueeze(-2) * k_block).sum(dim=-1)
            logits = logits.clamp(min=-max_logit, max=max_logit)
            log_p_choices = torch.stack(
                [F.logsigmoid(logits), F.logsigmoid(-logits)],
                dim=-1,
            )
            directions = torch.argmax(log_p_choices + gumbels[start:stop], dim=-1)
            chosen_log_probs = torch.gather(
                log_p_choices, -1, directions.unsqueeze(-1)
            ).squeeze(-1)
            accumulated_log_probs[start:stop] = (
                accumulated_log_probs[start:stop] + chosen_log_probs
            )
            packed_paths[start:stop, ..., depth // 8].bitwise_or_(
                directions.to(torch.uint8) << (depth % 8)
            )
            current_nodes[start:stop] = 2 * node_block + 1 + directions.to(torch.int32)

    current_nodes.sub_(k_len)
    return current_nodes, accumulated_log_probs, packed_paths


def _sample_paths_non_causal_streaming(
    q: torch.Tensor,
    k: torch.Tensor,
    num_samples: int,
    *,
    block_size: int,
    max_logit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        q.is_cuda
        and k.is_cuda
        and q.dtype == k.dtype
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and k.shape[0] > 0
    ):
        native_result = _native.sample_non_causal_paths_forward(
            q.contiguous(),
            k.contiguous(),
            num_samples,
            block_size,
            max_logit,
        )
        if native_result is not None:
            sampled_indices, path_log_probs, packed_paths = native_result
            _note_runtime_usage("sample", native=True)
            return sampled_indices, path_log_probs, packed_paths

    _note_runtime_usage("sample", native=False)
    return _sample_paths_non_causal_streaming_python(
        q,
        k,
        num_samples,
        block_size=block_size,
        max_logit=max_logit,
    )


def _replay_non_causal_paths_python(
    q: torch.Tensor,
    k: torch.Tensor,
    packed_paths: torch.Tensor,
    *,
    block_size: int,
    max_logit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct sampled leaves and log-probabilities from packed path bits."""
    l_q, batch, nheads, num_samples, _ = packed_paths.shape
    k_len = k.shape[0]
    num_leaves = k_len + 1
    if k_len == 0:
        sampled = torch.zeros(
            (l_q, batch, nheads, num_samples), device=q.device, dtype=torch.int64
        )
        log_probs = torch.zeros(
            (l_q, batch, nheads, num_samples),
            device=q.device,
            dtype=_accum_dtype(q.dtype),
        )
        return sampled, log_probs

    accum_dtype = _accum_dtype(q.dtype)
    log_n = num_leaves.bit_length() - 1
    current_nodes = torch.zeros(
        (l_q, batch, nheads, num_samples), device=q.device, dtype=torch.int32
    )
    accumulated_log_probs = torch.zeros(
        (l_q, batch, nheads, num_samples), device=q.device, dtype=accum_dtype
    )
    q_accum = q.to(accum_dtype)
    k_accum = k.to(accum_dtype)

    for depth in range(log_n):
        for start in range(0, l_q, block_size):
            stop = min(start + block_size, l_q)
            node_block = current_nodes[start:stop]
            q_block = q_accum[start:stop]
            k_block = _gather_tree_values(k_accum, node_block)
            logits = (q_block.unsqueeze(-2) * k_block).sum(dim=-1)
            logits = logits.clamp(min=-max_logit, max=max_logit)
            directions = _read_path_bit(packed_paths[start:stop], depth)
            chosen_log_probs = torch.where(
                directions == 0,
                F.logsigmoid(logits),
                F.logsigmoid(-logits),
            )
            accumulated_log_probs[start:stop] = (
                accumulated_log_probs[start:stop] + chosen_log_probs
            )
            current_nodes[start:stop] = 2 * node_block + 1 + directions

    current_nodes.sub_(k_len)
    return current_nodes, accumulated_log_probs


def _replay_non_causal_paths(
    q: torch.Tensor,
    k: torch.Tensor,
    packed_paths: torch.Tensor,
    *,
    block_size: int,
    max_logit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        q.is_cuda
        and k.is_cuda
        and packed_paths.is_cuda
        and q.dtype == k.dtype
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and k.shape[0] > 0
    ):
        native_result = _native.replay_non_causal_paths_forward(
            q.contiguous(),
            k.contiguous(),
            packed_paths.contiguous(),
            max_logit,
        )
        if native_result is not None:
            sampled_indices, path_log_probs = native_result
            _note_runtime_usage("replay", native=True)
            return sampled_indices, path_log_probs

    _note_runtime_usage("replay", native=False)
    return _replay_non_causal_paths_python(
        q,
        k,
        packed_paths,
        block_size=block_size,
        max_logit=max_logit,
    )


def _weighted_value_sum(
    v: torch.Tensor,
    sampled_indices: torch.Tensor,
    attn_weights: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    if v.is_cuda and sampled_indices.is_cuda and attn_weights.is_cuda:
        native_output = _native.weighted_value_sum_forward(
            v,
            sampled_indices.contiguous(),
            attn_weights.to(torch.float32).contiguous(),
        )
        if native_output is not None:
            _note_runtime_usage("weighted_value", native=True)
            return native_output

    l_q, batch, nheads, _ = sampled_indices.shape
    width = v.shape[-1]
    accum_dtype = _accum_dtype(v.dtype)
    output = torch.empty(
        (l_q, batch, nheads, width), device=v.device, dtype=accum_dtype
    )
    v_accum = v.to(accum_dtype)
    weights_accum = attn_weights.to(accum_dtype)
    for start in range(0, l_q, block_size):
        stop = min(start + block_size, l_q)
        gathered_values = _gather_tree_values(v_accum, sampled_indices[start:stop])
        output[start:stop] = (
            weights_accum[start:stop].unsqueeze(-1) * gathered_values
        ).sum(dim=-2)
    _note_runtime_usage("weighted_value", native=False)
    return output.to(v.dtype)


def _softmax_path_log_probs_inplace(path_log_probs: torch.Tensor) -> torch.Tensor:
    path_log_probs.sub_(torch.logsumexp(path_log_probs, dim=-1, keepdim=True))
    path_log_probs.exp_()
    return path_log_probs


def _scatter_weighted_grad_v(
    grad_output: torch.Tensor,
    sampled_indices: torch.Tensor,
    attn_weights: torch.Tensor,
    *,
    num_leaves: int,
) -> torch.Tensor | None:
    if (
        grad_output.is_cuda
        and sampled_indices.is_cuda
        and attn_weights.is_cuda
        and grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and attn_weights.dtype == torch.float32
    ):
        native_output = _native.scatter_weighted_grad_v_forward(
            grad_output.contiguous(),
            sampled_indices.contiguous(),
            attn_weights.contiguous(),
            num_leaves,
        )
        if native_output is not None:
            _note_runtime_usage("scatter_grad_v", native=True)
            return native_output
    _note_runtime_usage("scatter_grad_v", native=False)
    return None


def _compute_grad_logit_non_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    current_nodes: torch.Tensor,
    packed_paths: torch.Tensor,
    grad_log_probs: torch.Tensor,
    *,
    depth: int,
    max_logit: float,
) -> torch.Tensor | None:
    if (
        q.is_cuda
        and k.is_cuda
        and current_nodes.is_cuda
        and packed_paths.is_cuda
        and grad_log_probs.is_cuda
        and q.dtype == k.dtype
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and grad_log_probs.dtype == torch.float32
    ):
        native_output = _native.compute_grad_logit_non_causal_forward(
            q.contiguous(),
            k.contiguous(),
            current_nodes.contiguous(),
            packed_paths.contiguous(),
            grad_log_probs.contiguous(),
            depth,
            max_logit,
        )
        if native_output is not None:
            _note_runtime_usage("grad_logit", native=True)
            return native_output
    _note_runtime_usage("grad_logit", native=False)
    return None


def _accumulate_qk_non_causal_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    current_nodes: torch.Tensor,
    packed_paths: torch.Tensor,
    grad_log_probs: torch.Tensor,
    *,
    depth: int,
    max_logit: float,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
) -> bool:
    if (
        q.is_cuda
        and k.is_cuda
        and current_nodes.is_cuda
        and packed_paths.is_cuda
        and grad_log_probs.is_cuda
        and grad_q_out.is_cuda
        and grad_k_out.is_cuda
        and q.dtype == k.dtype
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and grad_log_probs.dtype == torch.float32
        and grad_q_out.dtype == torch.float32
        and grad_k_out.dtype == torch.float32
    ):
        used_native = _native.accumulate_qk_non_causal_inplace(
            q.contiguous(),
            k.contiguous(),
            current_nodes.contiguous(),
            packed_paths.contiguous(),
            grad_log_probs.contiguous(),
            depth,
            max_logit,
            grad_q_out,
            grad_k_out,
        )
        _note_runtime_usage("accumulate_qk", native=used_native)
        return used_native
    _note_runtime_usage("accumulate_qk", native=False)
    return False


def _accumulate_qk_non_causal_all_depths_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    packed_paths: torch.Tensor,
    grad_log_probs: torch.Tensor,
    *,
    max_logit: float,
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
) -> bool:
    if (
        q.is_cuda
        and k.is_cuda
        and packed_paths.is_cuda
        and grad_log_probs.is_cuda
        and grad_q_out.is_cuda
        and grad_k_out.is_cuda
        and q.dtype == k.dtype
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and grad_log_probs.dtype == torch.float32
        and grad_q_out.dtype == torch.float32
        and grad_k_out.dtype in (q.dtype, torch.float32)
    ):
        used_native = _native.accumulate_qk_non_causal_all_depths_inplace(
            q.contiguous(),
            k.contiguous(),
            packed_paths.contiguous(),
            grad_log_probs.contiguous(),
            max_logit,
            grad_q_out,
            grad_k_out,
        )
        _note_runtime_usage("accumulate_qk_all_depths", native=used_native)
        return used_native
    _note_runtime_usage("accumulate_qk_all_depths", native=False)
    return False


class _StochasticNonCausalTreeAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_samples: int,
        block_size: int,
        max_logit: float,
    ):
        num_leaves = v.shape[0]
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")

        if num_leaves == 0:
            output = torch.zeros_like(q)
            weights = torch.empty(
                (*q.shape[:-1], 0), device=q.device, dtype=q.dtype
            )
            ctx.mark_non_differentiable(weights)
            return output, weights

        if num_leaves == 1:
            output = v.expand(q.shape[0], -1, -1, -1).contiguous()
            weights = torch.ones(
                (*q.shape[:-1], 1), device=q.device, dtype=q.dtype
            )
            packed_paths = _empty_packed_paths(
                q.shape[:-1],
                num_samples=num_samples,
                num_leaves=num_leaves,
                device=q.device,
            )
            ctx.save_for_backward(q, k, v, packed_paths)
            ctx.block_size = block_size
            ctx.max_logit = max_logit
            ctx.mark_non_differentiable(weights)
            return output, weights

        sampled_indices, path_log_probs, packed_paths = _sample_paths_non_causal_streaming(
            q,
            k,
            num_samples,
            block_size=block_size,
            max_logit=max_logit,
        )
        attn_weights = _softmax_path_log_probs_inplace(path_log_probs)
        output = _weighted_value_sum(
            v,
            sampled_indices,
            attn_weights,
            block_size=block_size,
        )

        ctx.save_for_backward(q, k, v, packed_paths)
        ctx.block_size = block_size
        ctx.max_logit = max_logit
        ctx.mark_non_differentiable(attn_weights)
        return output, attn_weights

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_attn_weights: torch.Tensor | None):
        del grad_attn_weights
        q, k, v, packed_paths = ctx.saved_tensors
        block_size = ctx.block_size
        max_logit = ctx.max_logit

        num_leaves = v.shape[0]
        k_len = k.shape[0]
        accum_dtype = _accum_dtype(q.dtype)
        q_accum: torch.Tensor | None = None
        k_accum: torch.Tensor | None = None

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k, dtype=accum_dtype)
        grad_v = torch.zeros_like(v)

        def _ensure_qk_accum() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal q_accum, k_accum
            if q_accum is None:
                q_accum = q.to(accum_dtype)
            if k_accum is None:
                k_accum = k.to(accum_dtype)
            return q_accum, k_accum

        def _ensure_grad_v_accum() -> torch.Tensor:
            nonlocal grad_v
            if grad_v.dtype != accum_dtype:
                grad_v = grad_v.to(accum_dtype)
            return grad_v

        if num_leaves == 0:
            return grad_q, grad_k.to(k.dtype), grad_v.to(v.dtype), None, None, None

        if num_leaves == 1:
            _ensure_grad_v_accum()[0] = grad_output.to(accum_dtype).sum(dim=0)
            return grad_q, grad_k.to(k.dtype), grad_v.to(v.dtype), None, None, None

        sampled_indices, path_log_probs = _replay_non_causal_paths(
            q,
            k,
            packed_paths,
            block_size=block_size,
            max_logit=max_logit,
        )
        attn_weights = _softmax_path_log_probs_inplace(path_log_probs)
        grad_v_native = _scatter_weighted_grad_v(
            grad_output,
            sampled_indices,
            attn_weights,
            num_leaves=num_leaves,
        )
        if grad_v_native is not None:
            grad_v = grad_v_native
        log_n = num_leaves.bit_length() - 1
        for start in range(0, q.shape[0], block_size):
            stop = min(start + block_size, q.shape[0])
            q_block = q[start:stop]
            grad_out_block = grad_output[start:stop]
            grad_out_block_accum = grad_out_block.to(accum_dtype)
            idx_block = sampled_indices[start:stop]
            attn_block = attn_weights[start:stop]

            gathered_values = _gather_tree_values(v, idx_block).to(accum_dtype)
            grad_attn = (grad_out_block_accum.unsqueeze(-2) * gathered_values).sum(dim=-1)
            grad_log_probs = attn_block * (
                grad_attn - (attn_block * grad_attn).sum(dim=-1, keepdim=True)
            )

            if grad_v_native is None:
                grad_v_accum = _ensure_grad_v_accum()
                grad_v_updates = attn_block.unsqueeze(-1) * grad_out_block_accum.unsqueeze(-2)
                grad_v = grad_v_accum + _scatter_tree_updates(
                    num_leaves, idx_block, grad_v_updates
                )

            grad_q_block = torch.zeros_like(grad_out_block, dtype=accum_dtype)
            used_full_native_qk = _accumulate_qk_non_causal_all_depths_inplace(
                q_block,
                k,
                packed_paths[start:stop],
                grad_log_probs,
                max_logit=max_logit,
                grad_q_out=grad_q_block,
                grad_k_out=grad_k,
            )
            if not used_full_native_qk:
                q_accum_local, k_accum_local = _ensure_qk_accum()
                q_block_accum = q_accum_local[start:stop]
                node_idx = torch.zeros_like(idx_block)
                for depth in range(log_n):
                    bit = _read_path_bit(packed_paths[start:stop], depth)
                    used_native_qk = _accumulate_qk_non_causal_inplace(
                        q_block_accum,
                        k_accum_local,
                        node_idx,
                        packed_paths[start:stop],
                        grad_log_probs,
                        depth=depth,
                        max_logit=max_logit,
                        grad_q_out=grad_q_block,
                        grad_k_out=grad_k,
                    )
                    if not used_native_qk:
                        grad_logit = _compute_grad_logit_non_causal(
                            q_block_accum,
                            k_accum_local,
                            node_idx,
                            packed_paths[start:stop],
                            grad_log_probs,
                            depth=depth,
                            max_logit=max_logit,
                        )
                    if not used_native_qk and grad_logit is None:
                        branch_sign = torch.where(bit == 0, 1.0, -1.0).to(accum_dtype)
                        gathered_keys = _gather_tree_values(k_accum_local, node_idx)
                        logits = (q_block_accum.unsqueeze(-2) * gathered_keys).sum(dim=-1)
                        logits = logits.clamp(min=-max_logit, max=max_logit)
                        grad_logit = branch_sign * torch.sigmoid(-branch_sign * logits)
                        grad_logit = grad_logit * grad_log_probs

                        grad_q_block = grad_q_block + (
                            grad_logit.unsqueeze(-1) * gathered_keys
                        ).sum(dim=-2)
                        grad_k_updates = grad_logit.unsqueeze(-1) * q_block_accum.unsqueeze(-2)
                        grad_k = grad_k + _scatter_tree_updates(
                            k_len, node_idx, grad_k_updates
                        )
                    elif not used_native_qk:
                        grad_q_contrib = _weighted_value_sum(
                            k_accum_local,
                            node_idx,
                            grad_logit,
                            block_size=block_size,
                        )
                        grad_q_block = grad_q_block + grad_q_contrib
                        grad_k_contrib = _scatter_weighted_grad_v(
                            q_block_accum,
                            node_idx,
                            grad_logit,
                            num_leaves=k_len,
                        )
                        if grad_k_contrib is None:
                            raise RuntimeError(
                                "native grad_logit path requires native scatter kernel"
                            )
                        grad_k = grad_k + grad_k_contrib
                    if depth + 1 < log_n:
                        node_idx = 2 * node_idx + 1 + bit

            grad_q[start:stop] = grad_q_block.to(q.dtype)

        return grad_q, grad_k.to(k.dtype), grad_v.to(v.dtype), None, None, None


def hierarchical_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    is_causal: bool = False,
    mode: Mode = "stochastic",
    num_samples: int = 64,
    *,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    max_logit: float = _DEFAULT_MAX_LOGIT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical attention with a custom autograd path for non-causal stochastic mode."""
    if mode != "stochastic" or is_causal:
        return reference_hierarchical_attention(
            query,
            key,
            value,
            num_heads=num_heads,
            is_causal=is_causal,
            mode=mode,
            num_samples=num_samples,
        )

    if query.dim() != 3 or value.dim() != 3:
        raise ValueError(
            "query and value must have shape (L, B, E) and (N, B, E)"
        )

    l_q, batch, embed = query.shape
    num_leaves = value.shape[0]
    if embed % num_heads != 0:
        raise ValueError(
            f"embed_dim={embed} is not divisible by num_heads={num_heads}"
        )
    width = embed // num_heads

    q = query.reshape(l_q, batch, num_heads, width)
    if num_leaves > 1:
        k = key.reshape(num_leaves - 1, batch, num_heads, width)
    else:
        k = key.new_empty((0, batch, num_heads, width))
    v = value.reshape(num_leaves, batch, num_heads, width)

    output, attn_weights = _StochasticNonCausalTreeAttention.apply(
        q,
        k,
        v,
        num_samples,
        block_size,
        max_logit,
    )
    return output.reshape(l_q, batch, embed), attn_weights