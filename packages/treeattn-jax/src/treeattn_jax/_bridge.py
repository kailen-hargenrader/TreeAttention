from functools import lru_cache, partial
from typing import cast

import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from jax import dlpack as jdl
from torch.utils import dlpack as tdl

from treeattn_jax._tree_mha_jax import (
    query_causal_frontier_update,
    tree_attention_causal_stochastic,
    tree_attention_deterministic_causal,
    tree_attention_deterministic_noncausal,
    tree_attention_noncausal_stochastic,
)



@lru_cache(maxsize=None)
def _get_jitted_vjp_deterministic_noncausal(dropout, return_weights, max_logit):
    if dropout is None:

        @partial(jax.vmap, in_axes=(1, 1, 1), out_axes=1)  # vmap over batch
        @partial(jax.vmap, in_axes=(1, 1, 1), out_axes=1)  # vmap over head
        @partial(jax.vmap, in_axes=(0, None, None), out_axes=0)  # vmap over query
        def attn_fx_nodrop(q, k, v):
            return tree_attention_deterministic_noncausal(
                q,
                k,
                v,
                dropout=None,
                rng=None,
                return_weights=return_weights,
                max_logit=max_logit,
            )

        return jax.jit(partial(jax.vjp, attn_fx_nodrop))
    else:

        @partial(jax.vmap, in_axes=(1, 1, 1, 1), out_axes=1)  # vmap over batch
        @partial(jax.vmap, in_axes=(1, 1, 1, 1), out_axes=1)  # vmap over head
        @partial(jax.vmap, in_axes=(0, None, None, 0), out_axes=0)  # vmap over query
        def attn_fx_drop(q, k, v, rng):
            return tree_attention_deterministic_noncausal(
                q,
                k,
                v,
                rng=rng,
                dropout=dropout,
                return_weights=return_weights,
                max_logit=max_logit,
            )

        return jax.jit(partial(jax.vjp, attn_fx_drop))


class JAXTreeNonCausalDeterministicAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_torch: torch.Tensor,
        k_torch: torch.Tensor,
        v_torch: torch.Tensor,
        dropout: float | None,
        return_weights: bool,
        max_logit: float,
    ):
        """PyTorch-facing deterministic non-causal Tree MHA via JAX weights.

        Args:
                q: [L, B, H, D]
                k: [K, B, H, D]
                v: [N, B, H, D]
                return_weights: bool
                max_logit: float

        Returns:
                attn_output: [L, B, H, D]
                attn_weights: [L, N, B, H] if return_weights is True, otherwise None
        """
        q_torch = q_torch.detach().contiguous()
        k_torch = k_torch.detach().contiguous()
        v_torch = v_torch.detach().contiguous()

        # check that all tensors are on the same device
        assert q_torch.device == k_torch.device == v_torch.device, (
            "All tensors must be on the same device"
        )
        device = q_torch.device

        # Torch -> JAX zero-copy
        q_jax = jdl.from_dlpack(q_torch)
        k_jax = jdl.from_dlpack(k_torch)
        v_jax = jdl.from_dlpack(v_torch)

        attn_fx_vjp_jit = _get_jitted_vjp_deterministic_noncausal(
            dropout, return_weights, max_logit
        )

        if dropout is not None:
            # Generate a seed from PyTorch's RNG and use it to seed JAX's RNG.
            seed = torch.randint(jnp.iinfo(jnp.int32).max, (1,)).item()
            rng = jax.random.PRNGKey(seed)
            # Per (L,B,H) rngs
            rngs = jax.random.split(
                rng, (q_jax.shape[0], q_jax.shape[1], q_jax.shape[2])
            )
            jax_returns, f_vjp = attn_fx_vjp_jit(q_jax, k_jax, v_jax, rngs)
        else:
            jax_returns, f_vjp = attn_fx_vjp_jit(q_jax, k_jax, v_jax)
        ctx.f_vjp = f_vjp
        ctx.return_weights = return_weights
        if return_weights:
            attn_jax, weights_jax = jax_returns  # attn: [L,B,H,D], weights: [L,B,H,N]
            weights_jax = rearrange(weights_jax, "l b h n -> l n b h")
            attn_torch = tdl.from_dlpack(attn_jax)
            weights_torch = tdl.from_dlpack(weights_jax)
            return attn_torch, weights_torch
        else:
            attn_jax = jax_returns
            attn_torch = tdl.from_dlpack(attn_jax)
            return attn_torch

    @staticmethod
    def backward(ctx, *grad_outputs):
        f_vjp = ctx.f_vjp
        return_weights = ctx.return_weights

        if return_weights:
            grad_attn_torch, grad_weights_torch = grad_outputs
            grad_attn_torch = grad_attn_torch.contiguous()
            grad_attn_jax = jdl.from_dlpack(grad_attn_torch)

            assert grad_weights_torch is not None, (
                "Gradients for weights must be provided"
            )
            grad_weights_torch = grad_weights_torch.contiguous()
            grad_weights_jax = jdl.from_dlpack(grad_weights_torch)
            grad_weights_jax = rearrange(grad_weights_jax, "l n b h -> l b h n")

            g_jax = (grad_attn_jax, grad_weights_jax)
        else:
            (grad_attn_torch,) = grad_outputs
            grad_attn_torch = grad_attn_torch.contiguous()
            g_jax = jdl.from_dlpack(grad_attn_torch)

        (dq_jax, dk_jax, dv_jax) = f_vjp(g_jax)

        # Block until JAX computation completes (important for CUDA)
        dq_jax = dq_jax.block_until_ready()
        dk_jax = dk_jax.block_until_ready()
        dv_jax = dv_jax.block_until_ready()

        dq_torch = tdl.from_dlpack(dq_jax)
        dk_torch = tdl.from_dlpack(dk_jax)
        dv_torch = tdl.from_dlpack(dv_jax)

        return dq_torch, dk_torch, dv_torch, None, None, None


@lru_cache(maxsize=None)
def _get_jitted_vjp_deterministic_causal(dropout, return_weights, max_logit):
    if dropout is None:

        @partial(jax.vmap, in_axes=(1, 1, 1), out_axes=1)  # vmap over batch
        @partial(jax.vmap, in_axes=(1, 1, 1), out_axes=1)  # vmap over head
        def attn_fx_nodrop(q, k, v):
            return tree_attention_deterministic_causal(
                q,
                k,
                v,
                rng=None,
                dropout=None,
                return_weights=return_weights,
                max_logit=max_logit,
            )

        return jax.jit(partial(jax.vjp, attn_fx_nodrop))
    else:

        @partial(jax.vmap, in_axes=(1, 1, 1, 1), out_axes=1)  # vmap over batch
        @partial(jax.vmap, in_axes=(1, 1, 1, 1), out_axes=1)  # vmap over head
        def attn_fx_drop(q, k, v, rng):
            return tree_attention_deterministic_causal(
                q,
                k,
                v,
                rng=rng,
                dropout=dropout,
                return_weights=return_weights,
                max_logit=max_logit,
            )

        return jax.jit(partial(jax.vjp, attn_fx_drop))


class JAXTreeCausalDeterministicAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_torch: torch.Tensor,
        k_torch: torch.Tensor,
        v_torch: torch.Tensor,
        dropout: float | None,
        return_weights: bool,
        max_logit: float,
    ):
        """PyTorch-facing deterministic non-causal Tree MHA via JAX weights.

        Args:
                q: [L, B, H, D]
                k: [K, B, H, D]
                v: [N, B, H, D]
                return_weights: bool
                max_logit: float

        Returns:
                attn_output: [L, B, H, D]
                attn_weights: [L, N, B, H] if return_weights is True, otherwise None
        """
        q_torch = q_torch.detach().contiguous()
        k_torch = k_torch.detach().contiguous()
        v_torch = v_torch.detach().contiguous()

        # check that all tensors are on the same device
        assert q_torch.device == k_torch.device == v_torch.device, (
            "All tensors must be on the same device"
        )
        device = q_torch.device

        # Torch -> JAX zero-copy
        q_jax = jdl.from_dlpack(q_torch)
        k_jax = jdl.from_dlpack(k_torch)
        v_jax = jdl.from_dlpack(v_torch)

        attn_fx_vjp_jit = _get_jitted_vjp_deterministic_causal(
            dropout, return_weights, max_logit
        )

        if dropout is not None:
            seed = torch.randint(jnp.iinfo(jnp.int32).max, (1,)).item()
            rng = jax.random.PRNGKey(seed)
            rngs = jax.random.split(
                rng, (q_jax.shape[0], q_jax.shape[1], q_jax.shape[2])
            )
            jax_returns, f_vjp = attn_fx_vjp_jit(q_jax, k_jax, v_jax, rngs)
        else:
            jax_returns, f_vjp = attn_fx_vjp_jit(q_jax, k_jax, v_jax)
        ctx.f_vjp = f_vjp
        ctx.return_weights = return_weights
        if return_weights:
            attn_jax, weights_jax = jax_returns  # attn: [L,B,H,D], weights: [L,B,H,N]
            weights_jax = rearrange(weights_jax, "l b h n -> l n b h")
            attn_torch = tdl.from_dlpack(attn_jax)
            weights_torch = tdl.from_dlpack(weights_jax)
            return attn_torch, weights_torch
        else:
            attn_jax = jax_returns
            attn_torch = tdl.from_dlpack(attn_jax)
            return attn_torch

    @staticmethod
    def backward(ctx, *grad_outputs):
        f_vjp = ctx.f_vjp
        return_weights = ctx.return_weights

        if return_weights:
            grad_attn_torch, grad_weights_torch = grad_outputs
            grad_attn_torch = grad_attn_torch.contiguous()
            grad_attn_jax = jdl.from_dlpack(grad_attn_torch)

            assert grad_weights_torch is not None, (
                "Gradients for weights must be provided"
            )
            grad_weights_torch = grad_weights_torch.contiguous()
            grad_weights_jax = jdl.from_dlpack(grad_weights_torch)
            grad_weights_jax = rearrange(grad_weights_jax, "l n b h -> l b h n")

            g_jax = (grad_attn_jax, grad_weights_jax)
        else:
            (grad_attn_torch,) = grad_outputs
            grad_attn_torch = grad_attn_torch.contiguous()
            g_jax = jdl.from_dlpack(grad_attn_torch)

        (dq_jax, dk_jax, dv_jax) = f_vjp(g_jax)

        # Block until JAX computation completes (important for CUDA)
        dq_jax = dq_jax.block_until_ready()
        dk_jax = dk_jax.block_until_ready()
        dv_jax = dv_jax.block_until_ready()

        dq_torch = tdl.from_dlpack(dq_jax)
        dk_torch = tdl.from_dlpack(dk_jax)
        dv_torch = tdl.from_dlpack(dv_jax)

        return dq_torch, dk_torch, dv_torch, None, None, None


@lru_cache(maxsize=None)
def _get_jitted_vjp_stochastic_noncausal(
    num_samples, dropout, return_weights, gumbel_scale, max_logit
):
    @partial(jax.vmap, in_axes=(1, 1, 1, 1), out_axes=1)  # vmap over batch
    @partial(jax.vmap, in_axes=(1, 1, 1, 1), out_axes=1)  # vmap over head
    @partial(jax.vmap, in_axes=(0, None, None, 0), out_axes=0)  # vmap over query
    def attn_fx(q, k, v, rng):
        return tree_attention_noncausal_stochastic(
            q,
            k,
            v,
            num_samples=num_samples,
            rng=rng,
            dropout=dropout,
            gumbel_scale=gumbel_scale,
            return_weights=return_weights,
            max_logit=max_logit,
        )

    return jax.jit(partial(jax.vjp, attn_fx))


class JAXTreeNonCausalStochasticAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_torch: torch.Tensor,
        k_torch: torch.Tensor,
        v_torch: torch.Tensor,
        num_samples: int,
        dropout: float | None,
        return_weights: bool,
        gumbel_scale: float,
        max_logit: float,
    ):
        """PyTorch-facing deterministic non-causal Tree MHA via JAX weights.

        Args:
                q: [L, B, H, D]
                k: [K, B, H, D]
                v: [N, B, H, D]
                return_weights: bool
                max_logit: float

        Returns:
                attn_output: [L, B, H, D]
                attn_weights: [L, N, B, H] if return_weights is True, otherwise None
        """
        # Generate a seed from PyTorch's RNG and use it to seed JAX's RNG.
        # This ensures that setting torch.manual_seed() makes the JAX part deterministic.
        seed = torch.randint(jnp.iinfo(jnp.int32).max, (1,)).item()
        rng = jax.random.PRNGKey(seed)
        q_torch = q_torch.detach().contiguous()
        k_torch = k_torch.detach().contiguous()
        v_torch = v_torch.detach().contiguous()

        # check that all tensors are on the same device
        assert q_torch.device == k_torch.device == v_torch.device, (
            "All tensors must be on the same device"
        )
        device = q_torch.device

        # Torch -> JAX zero-copy
        q_jax = jdl.from_dlpack(q_torch)
        k_jax = jdl.from_dlpack(k_torch)
        v_jax = jdl.from_dlpack(v_torch)

        attn_fx_vjp_jit = _get_jitted_vjp_stochastic_noncausal(
            num_samples, dropout, return_weights, gumbel_scale, max_logit
        )
        rngs = jax.random.split(rng, (q_jax.shape[0], q_jax.shape[1], q_jax.shape[2]))
        jax_returns, f_vjp = attn_fx_vjp_jit(q_jax, k_jax, v_jax, rngs)
        ctx.f_vjp = f_vjp
        ctx.return_weights = return_weights
        if return_weights:
            attn_jax, (weights_jax, idxs_jax) = (
                jax_returns  # attn: [L,B,H,D], weights: [L,B,H,N]
            )
            weights_jax = rearrange(weights_jax, "l b h n -> l n b h")
            idxs_jax = rearrange(idxs_jax, "l b h n -> l n b h")
            attn_torch = tdl.from_dlpack(attn_jax)
            weights_torch = tdl.from_dlpack(weights_jax)
            idxs_torch = tdl.from_dlpack(idxs_jax)
            idxs_torch = idxs_torch.to(torch.int64)
            return attn_torch, weights_torch, idxs_torch
        else:
            attn_jax = jax_returns
            attn_torch = tdl.from_dlpack(attn_jax)
            return attn_torch

    @staticmethod
    def backward(ctx, *grad_outputs):
        f_vjp = ctx.f_vjp
        return_weights = ctx.return_weights

        if return_weights:
            grad_attn_torch, grad_weights_torch, grad_idxs_torch = grad_outputs
            grad_attn_torch = grad_attn_torch.contiguous()
            grad_attn_jax = jdl.from_dlpack(grad_attn_torch)

            assert grad_weights_torch is not None, (
                "Gradients for weights must be provided"
            )
            assert grad_idxs_torch is not None, "Gradients for indices must be provided"
            grad_weights_torch = grad_weights_torch.contiguous()
            grad_weights_jax = jdl.from_dlpack(grad_weights_torch)
            grad_weights_jax = rearrange(grad_weights_jax, "l n b h -> l b h n")

            grad_idxs_torch = grad_idxs_torch.contiguous()
            grad_idxs_jax = jdl.from_dlpack(grad_idxs_torch)
            grad_idxs_jax = rearrange(grad_idxs_jax, "l n b h -> l b h n")

            g_jax = (grad_attn_jax, (grad_weights_jax, grad_idxs_jax))
        else:
            (grad_attn_torch,) = grad_outputs
            grad_attn_torch = grad_attn_torch.contiguous()
            g_jax = jdl.from_dlpack(grad_attn_torch)

        (dq_jax, dk_jax, dv_jax, _) = f_vjp(g_jax)  # last argument is for the rng

        # Block until JAX computation completes (important for CUDA)
        dq_jax = dq_jax.block_until_ready()
        dk_jax = dk_jax.block_until_ready()
        dv_jax = dv_jax.block_until_ready()

        dq_torch = tdl.from_dlpack(dq_jax)
        dk_torch = tdl.from_dlpack(dk_jax)
        dv_torch = tdl.from_dlpack(dv_jax)

        return dq_torch, dk_torch, dv_torch, None, None, None, None, None


@lru_cache(maxsize=None)
def _get_jitted_vjp_stochastic_causal(
    num_samples, dropout, return_weights, gumbel_scale, max_logit
):
    @partial(jax.vmap, in_axes=(1, 1, 1, 1), out_axes=1)  # vmap over batch
    @partial(jax.vmap, in_axes=(1, 1, 1, 1), out_axes=1)  # vmap over head
    def attn_fx(q, k, v, rng):
        # unlike the non-causal case, we cannot vmap over queries
        return tree_attention_causal_stochastic(
            q,
            k,
            v,
            num_samples=num_samples,
            rng=rng,
            dropout=dropout,
            gumbel_scale=gumbel_scale,
            return_weights=return_weights,
            max_logit=max_logit,
        )

    return jax.jit(partial(jax.vjp, attn_fx))


class JAXTreeCausalStochasticAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_torch: torch.Tensor,
        k_torch: torch.Tensor,
        v_torch: torch.Tensor,
        num_samples: int,
        dropout: float | None,
        return_weights: bool,
        gumbel_scale: float,
        max_logit: float,
    ):
        """PyTorch-facing deterministic non-causal Tree MHA via JAX weights.

        Args:
                q: [L, B, H, D]
                k: [K, B, H, D]
                v: [N, B, H, D]
                return_weights: bool
                max_logit: float

        Returns:
                attn_output: [L, B, H, D]
                attn_weights: [L, num_samples, B, H] if return_weights is True, otherwise None
                attn_idxs: [L, num_samples, B, H] if return_weights is True, otherwise None
        """
        # Generate a seed from PyTorch's RNG and use it to seed JAX's RNG.
        # This ensures that setting torch.manual_seed() makes the JAX part deterministic.
        seed = torch.randint(jnp.iinfo(jnp.int32).max, (1,)).item()
        rng = jax.random.PRNGKey(seed)
        q_torch = q_torch.detach().contiguous()
        k_torch = k_torch.detach().contiguous()
        v_torch = v_torch.detach().contiguous()

        # check that all tensors are on the same device
        assert q_torch.device == k_torch.device == v_torch.device, (
            "All tensors must be on the same device"
        )
        device = q_torch.device

        # Torch -> JAX zero-copy
        q_jax = jdl.from_dlpack(q_torch)
        k_jax = jdl.from_dlpack(k_torch)
        v_jax = jdl.from_dlpack(v_torch)

        attn_fx_vjp_jit = _get_jitted_vjp_stochastic_causal(
            num_samples, dropout, return_weights, gumbel_scale, max_logit
        )
        rngs = jax.random.split(rng, (q_jax.shape[0], q_jax.shape[1], q_jax.shape[2]))
        jax_returns, f_vjp = attn_fx_vjp_jit(q_jax, k_jax, v_jax, rngs)
        ctx.f_vjp = f_vjp
        ctx.return_weights = return_weights
        if return_weights:
            attn_jax, (weights_jax, idxs_jax) = (
                jax_returns  # attn: [L,B,H,D], weights: [L,B,H,N]
            )
            weights_jax = rearrange(weights_jax, "l b h n -> l n b h")
            idxs_jax = rearrange(idxs_jax, "l b h n -> l n b h")
            attn_torch = tdl.from_dlpack(attn_jax)
            weights_torch = tdl.from_dlpack(weights_jax)
            idxs_torch = tdl.from_dlpack(idxs_jax)
            idxs_torch = idxs_torch.to(torch.int64)
            return attn_torch, weights_torch, idxs_torch
        else:
            attn_jax = jax_returns
            attn_torch = tdl.from_dlpack(attn_jax)
            return attn_torch

    @staticmethod
    def backward(ctx, *grad_outputs):
        f_vjp = ctx.f_vjp
        return_weights = ctx.return_weights

        if return_weights:
            grad_attn_torch, grad_weights_torch, grad_idxs_torch = grad_outputs
            grad_attn_torch = grad_attn_torch.contiguous()
            grad_attn_jax = jdl.from_dlpack(grad_attn_torch)

            assert grad_weights_torch is not None, (
                "Gradients for weights must be provided"
            )
            assert grad_idxs_torch is not None, "Gradients for indices must be provided"
            grad_weights_torch = grad_weights_torch.contiguous()
            grad_weights_jax = jdl.from_dlpack(grad_weights_torch)
            grad_weights_jax = rearrange(grad_weights_jax, "l n b h -> l b h n")

            grad_idxs_torch = grad_idxs_torch.contiguous()
            grad_idxs_jax = jdl.from_dlpack(grad_idxs_torch)
            grad_idxs_jax = rearrange(grad_idxs_jax, "l n b h -> l b h n")

            g_jax = (grad_attn_jax, (grad_weights_jax, grad_idxs_jax))
        else:
            (grad_attn_torch,) = grad_outputs
            grad_attn_torch = grad_attn_torch.contiguous()
            g_jax = jdl.from_dlpack(grad_attn_torch)

        (dq_jax, dk_jax, dv_jax, _) = f_vjp(g_jax)  # last argument is for the rng

        # Block until JAX computation completes (important for CUDA)
        dq_jax = dq_jax.block_until_ready()
        dk_jax = dk_jax.block_until_ready()
        dv_jax = dv_jax.block_until_ready()

        dq_torch = tdl.from_dlpack(dq_jax)
        dk_torch = tdl.from_dlpack(dk_jax)
        dv_torch = tdl.from_dlpack(dv_jax)

        return dq_torch, dk_torch, dv_torch, None, None, None, None, None


def jax_tree_attention_noncausal_deterministic(
    q, k, v, dropout: float | None = None, return_weights=False, max_logit=1e3
):
    """JAX-based implementation of deterministic non-causal tree attention.

    Args:
            q: [L, B, H, D]
            k: [K, B, H, D]
            v: [N, B, H, D]
            return_weights: bool
            max_logit: float
    """
    return JAXTreeNonCausalDeterministicAttention.apply(
        q, k, v, dropout, return_weights, max_logit
    )


def jax_tree_attention_causal_deterministic(
    q, k, v, dropout: float | None = None, return_weights=False, max_logit=1e3
):
    """JAX-based implementation of deterministic causal tree attention.

    Args:
            q: [L, B, H, D]
            k: [K, B, H, D]
            v: [N, B, H, D]
            return_weights: bool
            max_logit: float
    """
    return JAXTreeCausalDeterministicAttention.apply(
        q, k, v, dropout, return_weights, max_logit
    )


def jax_tree_attention_noncausal_stochastic(
    q,
    k,
    v,
    num_samples,
    dropout: float | None = None,
    return_weights=False,
    gumbel_scale=1.0,
    max_logit=1e3,
):
    """JAX-based implementation of stochastic non-causal tree attention.

    Args:
            q: [L, B, H, D]
            k: [K, B, H, D]
            v: [N, B, H, D]
            num_samples: int
            return_weights: bool
            max_logit: float
    """
    return JAXTreeNonCausalStochasticAttention.apply(
        q, k, v, num_samples, dropout, return_weights, gumbel_scale, max_logit
    )


def jax_tree_attention_causal_stochastic(
    q,
    k,
    v,
    num_samples,
    dropout: float | None = None,
    return_weights=False,
    gumbel_scale=1.0,
    max_logit=1e3,
):
    """JAX-based implementation of stochastic causal tree attention.

    Args:
            q: [L, B, H, D]
            k: [K, B, H, D]
            v: [N, B, H, D]
            num_samples: int
            return_weights: bool
            max_logit: float
    """
    return JAXTreeCausalStochasticAttention.apply(
        q, k, v, num_samples, dropout, return_weights, gumbel_scale, max_logit
    )

