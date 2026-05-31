from __future__ import annotations

import os
from pathlib import Path


_BUILD_ON_DEMAND_ENV = "TREEATTN_CUDA_BUILD"
_VERBOSE_BUILD_ENV = "TREEATTN_CUDA_VERBOSE"
_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if os.environ.get(_BUILD_ON_DEMAND_ENV, "").strip() != "1":
        return None

    if _EXTENSION is not None:
        return _EXTENSION

    import torch
    from torch.utils.cpp_extension import load

    # This package is targeting A100 first; default to SM80 unless the user
    # explicitly overrides the architecture list.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

    package_root = Path(__file__).resolve().parents[2]
    csrc_dir = package_root / "csrc"
    sources = [
        str(csrc_dir / "accumulate_qk_non_causal_all_depths.cu"),
        str(csrc_dir / "accumulate_qk_non_causal.cu"),
        str(csrc_dir / "compute_grad_logit_non_causal.cu"),
        str(csrc_dir / "scatter_weighted_grad_v.cu"),
        str(csrc_dir / "sample_non_causal_paths.cu"),
        str(csrc_dir / "treeattn_cuda.cpp"),
        str(csrc_dir / "replay_non_causal_paths.cu"),
        str(csrc_dir / "weighted_value_sum.cu"),
    ]
    _EXTENSION = load(
        name="treeattn_cuda_ext",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=torch.cuda.is_available(),
        verbose=os.environ.get(_VERBOSE_BUILD_ENV, "").strip() == "1",
    )
    return _EXTENSION


def has_native_kernels() -> bool:
    extension = _load_extension()
    return extension is not None and hasattr(
        extension, "accumulate_qk_non_causal_all_depths_inplace"
    ) and hasattr(
        extension, "accumulate_qk_non_causal_inplace"
    ) and hasattr(
        extension, "compute_grad_logit_non_causal_forward"
    ) and hasattr(
        extension, "sample_non_causal_paths_forward"
    ) and hasattr(
        extension, "scatter_weighted_grad_v_forward"
    ) and hasattr(
        extension, "weighted_value_sum_forward"
    ) and hasattr(extension, "replay_non_causal_paths_forward")


def weighted_value_sum_forward(v, sampled_indices, attn_weights):
    extension = _load_extension()
    if extension is None:
        return None
    return extension.weighted_value_sum_forward(v, sampled_indices, attn_weights)


def replay_non_causal_paths_forward(q, k, packed_paths, max_logit):
    extension = _load_extension()
    if extension is None:
        return None
    return extension.replay_non_causal_paths_forward(q, k, packed_paths, max_logit)


def sample_non_causal_paths_forward(q, k, num_samples, block_size, max_logit):
    extension = _load_extension()
    if extension is None:
        return None
    return extension.sample_non_causal_paths_forward(
        q,
        k,
        num_samples,
        block_size,
        max_logit,
    )


def scatter_weighted_grad_v_forward(grad_output, sampled_indices, attn_weights, num_leaves):
    extension = _load_extension()
    if extension is None:
        return None
    return extension.scatter_weighted_grad_v_forward(
        grad_output,
        sampled_indices,
        attn_weights,
        num_leaves,
    )


def compute_grad_logit_non_causal_forward(
    q,
    k,
    current_nodes,
    packed_paths,
    grad_log_probs,
    depth,
    max_logit,
):
    extension = _load_extension()
    if extension is None:
        return None
    return extension.compute_grad_logit_non_causal_forward(
        q,
        k,
        current_nodes,
        packed_paths,
        grad_log_probs,
        depth,
        max_logit,
    )


def accumulate_qk_non_causal_inplace(
    q,
    k,
    current_nodes,
    packed_paths,
    grad_log_probs,
    depth,
    max_logit,
    grad_q_out,
    grad_k_out,
):
    extension = _load_extension()
    if extension is None:
        return False
    extension.accumulate_qk_non_causal_inplace(
        q,
        k,
        current_nodes,
        packed_paths,
        grad_log_probs,
        depth,
        max_logit,
        grad_q_out,
        grad_k_out,
    )
    return True


def accumulate_qk_non_causal_all_depths_inplace(
    q,
    k,
    packed_paths,
    grad_log_probs,
    max_logit,
    grad_q_out,
    grad_k_out,
):
    extension = _load_extension()
    if extension is None:
        return False
    extension.accumulate_qk_non_causal_all_depths_inplace(
        q,
        k,
        packed_paths,
        grad_log_probs,
        max_logit,
        grad_q_out,
        grad_k_out,
    )
    return True