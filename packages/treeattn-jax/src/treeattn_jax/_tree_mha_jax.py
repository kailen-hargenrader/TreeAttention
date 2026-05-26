import jax
import jax.numpy as jnp
from jax.nn import log_sigmoid
from jax.random import gumbel
from functools import partial
import equinox as eqx


def tree_attention_deterministic_weights(
    q: jnp.ndarray,
    k: jnp.ndarray,
    causal_frontier_logits: jnp.ndarray | None = None,
    causal_frontier_idxs: jnp.ndarray | None = None,
    q_idx: jnp.ndarray | None = None,
    max_logit: float = 1e3,
) -> jnp.ndarray:
    """Deterministic non-causal tree attention weights (matches PyTorch reference).

    Args:
            q: [D]
            k: [K, D] where K = N-1 and N is number of leaves
            max_logit: clamp for stability

    Returns:
            attn_weights: [N, D]
    """
    assert (
        causal_frontier_logits is None
        and causal_frontier_idxs is None
        and q_idx is None
    ) or (
        causal_frontier_logits is not None
        and causal_frontier_idxs is not None
        and q_idx is not None
    ), (
        "causal_frontier_logits and causal_frontier_idxs must be either both None or both not None"
    )
    K = k.shape[0]
    N = K + 1
    if K < 0:
        raise ValueError("invalid K")
    if K == 0:
        # Single leaf
        return jnp.array([1.0], dtype=q.dtype)

    log_N = int(K).bit_length()

    logits = jnp.dot(q, k.T)  # K x D x D

    logits = jnp.clip(logits, -max_logit, max_logit)
    leaf_indices = jnp.arange(N, dtype=jnp.int32)

    def body(carry, top_down_lvl_idx):
        cumulative_logsig = carry

        # For each leaf, find its parent key in the current tree level
        # and determine if it's a left or right child. This is vectorized over all N leaves.
        start = 1 << top_down_lvl_idx
        step = start << 1

        # Parent key index for each leaf
        k_idx_for_leaf = (leaf_indices // step) * step + start - 1

        # Left (0) or right (1) child bit
        bit = (leaf_indices >> top_down_lvl_idx) & 1
        # Create a mask for valid parent keys (k_idx must be in [0, K-1])
        valid_mask = k_idx_for_leaf < K

        if causal_frontier_logits is not None and causal_frontier_idxs is not None:
            # if current_sample_idx is in causal frontier, use the causal frontier logits
            # otherwise compute logits from <q,K>
            layer_causal_key_idx = causal_frontier_idxs[top_down_lvl_idx]
            layer_causal_key_logit = causal_frontier_logits[top_down_lvl_idx]
            # Safely gather the pre-computed logits
            safe_k_idxs = jnp.clip(k_idx_for_leaf, 0, K - 1)
            non_causal_leaf_logit = logits[safe_k_idxs]
            logits_for_leaf = jnp.where(
                safe_k_idxs == layer_causal_key_idx,
                layer_causal_key_logit,
                non_causal_leaf_logit,
            )
            ### [DEBUG] Callback
            # jax.debug.callback(callback, **locals(), ordered=True)
        else:
            safe_k_idxs = jnp.clip(k_idx_for_leaf, 0, K - 1)
            logits_for_leaf = logits[safe_k_idxs]

        # Apply sign: -logit for left branch, +logit for right branch
        signed_dots = jnp.where(bit == 0, -logits_for_leaf, logits_for_leaf)
        signed_dots = jnp.clip(signed_dots, -max_logit, max_logit)

        # Compute log-sigmoid
        layer_logsig = jax.nn.log_sigmoid(signed_dots)

        # Zero-out contributions from invalid parents
        layer_logsig = jnp.where(valid_mask, layer_logsig, 0.0)

        return cumulative_logsig + layer_logsig, None

    cumulative_logsig = jnp.zeros((N,), dtype=q.dtype)
    cumulative_logsig, _ = jax.lax.scan(
        body, cumulative_logsig, jnp.arange(log_N, dtype=jnp.int32)
    )

    attn = jnp.exp(cumulative_logsig)

    return attn


def tree_attention_deterministic_noncausal(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    return_weights: bool = False,
    rng: jax.Array | None = None,
    dropout: float | None = None,
    max_logit: float = 1e3,
) -> tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray:
    """Deterministic non-causal tree attention weights (matches PyTorch reference).

    Args:
            q: [D]
            k: [K, D] where K = N-1 and N is number of leaves
            v: [N, D]
            max_logit: clamp for stability

    Returns:
            attn_weights: [N, D]
    """
    assert (rng is None) == (dropout is None), (
        "rng and dropout must both be None or both be provided"
    )
    weights = tree_attention_deterministic_weights(q, k, max_logit=max_logit)
    if dropout is not None:
        dropout_layer = eqx.nn.Dropout(p=dropout, inference=False)
        weights = dropout_layer(weights, key=rng)
    attn = jnp.einsum("n, n d -> d", weights, v)
    if return_weights:
        return attn, weights
    else:
        return attn


def tree_attention_deterministic_causal(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    return_weights: bool = False,
    rng: jax.Array | None = None,
    dropout: float | None = None,
    max_logit: float = 1e3,
) -> tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray:
    """Deterministic causal tree attention weights (matches PyTorch reference).

    Args:
            q: [L, D]
            k: [K, D] where K = N-1 and N is number of leaves
            v: [N, D]
            max_logit: clamp for stability

    Returns:
            attn_weights: [N, D]
    """
    assert (rng is None) == (dropout is None), (
        "rng and dropout must both be None or both be provided"
    )
    q_idxs = jnp.arange(q.shape[0], dtype=jnp.int32)
    query_batched_query_causal_frontier_update = jax.vmap(
        lambda *args: query_causal_frontier_update(*args, max_logit=max_logit),
        in_axes=(0, None, 0),
    )
    frontier_node_logits, frontier_node_idxs = (
        query_batched_query_causal_frontier_update(q, k, q_idxs)
    )
    query_batched_tree_deterministic_weights = jax.vmap(
        lambda *args: tree_attention_deterministic_weights(*args, max_logit=max_logit),
        in_axes=(0, None, 0, 0, 0),
    )
    weights = query_batched_tree_deterministic_weights(
        q, k, frontier_node_logits, frontier_node_idxs, q_idxs
    )
    if dropout is not None:
        dropout_layer = eqx.nn.Dropout(p=dropout, inference=False)
        weights = jax.vmap(lambda x, key: dropout_layer(x, key=key))(weights, rng)
    #### [DEBUG] Callback
    # jax.debug.callback(callback, **locals(), ordered=True)
    attn = jnp.einsum("l n, n d -> l d", weights, v)
    if return_weights:
        return attn, weights
    else:
        return attn


#### logit manipulation utilities for causal frontier update


def logit_sum_sigmoids(x, y):
    """Computes logit(σ(x) + σ(y))

    Args:
            x: [D]
            y: [D]

    Returns:
            logit_sum: [D]
    """
    t = x + y  # must be < 0 so that σ(x)+σ(y) < 1
    LOG2 = jnp.log(2.0)

    num = jax.nn.logsumexp(jnp.stack([x, y, t + LOG2]), axis=0)
    den = log1mexp(t)
    return num - den


def logit_product_sigmoids(x, y):
    """Computes logit(σ(x) * σ(y))

    Args:
            x: [D]
            y: [D]
    """
    lse = jax.nn.logsumexp(jnp.stack([jnp.zeros_like(x), x, y]), axis=0)
    result = x + y - lse
    return result


def log1mexp(x: jnp.ndarray) -> jnp.ndarray:
    r"""(COPY PASTED FROM JAX SRC) Numerically stable calculation of :math:`\log(1 - \exp(-x))`.

    Args:
      x: [D]
      max_logit: clamp for stability
    """
    LOG2 = jnp.log(2.0)
    x = jnp.clip(x, max=-1e-6)
    return jnp.where(x < -LOG2, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


def callback(**kwargs):
    # if kwargs["top_down_lvl_idx"].item() == 3:
    if kwargs["q_idx"].item() == 7:
        print(kwargs)
        # jax.debug.callback(callback, **locals(), ordered=True)


def query_causal_frontier_update(
    q: jnp.ndarray, k: jnp.ndarray, q_idx: jnp.ndarray, max_logit: float = 1e3
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the causal frontier update for a query.

    Args:
            q: [D]
            k: [K, D] where K = N-1 and N is number of leaves
            q_idx: [0]

    Returns:
            frontier node logits: [D]
            frontier node index: [0]
    """
    K = k.shape[0]
    log_N = int(K).bit_length()
    frontier_node_logits = jnp.empty((log_N,), dtype=q.dtype)
    frontier_node_idxs = jnp.empty((log_N,), dtype=jnp.int32)
    remaining_mass = jnp.array(max_logit, dtype=q.dtype)

    def body(carry, layer_idx):
        frontier_node_logits, frontier_node_idxs, remaining_mass = carry
        start = (1 << layer_idx) - 1
        step = 1 << (layer_idx + 1)
        frontier_node_idx = (q_idx // step) * step + start
        frontier_node_is_invalid = frontier_node_idx >= K
        frontier_node_idxs = jax.lax.cond(
            frontier_node_is_invalid,
            lambda: frontier_node_idxs.at[layer_idx].set(-1),
            lambda: frontier_node_idxs.at[layer_idx].set(frontier_node_idx),
        )
        frontier_node_idx = (~frontier_node_is_invalid) * frontier_node_idx
        from_left = q_idx <= frontier_node_idx
        frontier_node_non_causal_logit = jnp.dot(q, k[frontier_node_idx])
        frontier_node_non_causal_logit = jnp.clip(
            frontier_node_non_causal_logit, -max_logit, max_logit
        )
        remaining_mass = jnp.clip(remaining_mass, -max_logit, max_logit)
        frontier_node_non_causal_logit = jnp.clip(
            frontier_node_non_causal_logit, -max_logit, max_logit
        )

        def left_branch():
            new_frontier_logit = jnp.array(-jnp.inf, dtype=q.dtype)
            new_remaining_mass = logit_product_sigmoids(
                frontier_node_non_causal_logit, remaining_mass
            )
            return new_frontier_logit, new_remaining_mass

        def right_branch():
            new_right = logit_product_sigmoids(
                frontier_node_non_causal_logit, remaining_mass
            )
            new_frontier_logit = jax.nn.softplus(
                frontier_node_non_causal_logit
            ) - jax.nn.softplus(-new_right)
            new_remaining_mass = logit_sum_sigmoids(
                -frontier_node_non_causal_logit, new_right
            )
            return new_frontier_logit, new_remaining_mass

        new_frontier_logit, new_remaining_mass = jax.lax.cond(
            from_left, left_branch, right_branch
        )
        frontier_node_logits = frontier_node_logits.at[layer_idx].set(
            new_frontier_logit
        )
        remaining_mass = jax.lax.cond(
            frontier_node_is_invalid, lambda: remaining_mass, lambda: new_remaining_mass
        )
        return (frontier_node_logits, frontier_node_idxs, remaining_mass), None

    (frontier_node_logits, frontier_node_idxs, remaining_mass), _ = jax.lax.scan(
        body,
        (frontier_node_logits, frontier_node_idxs, remaining_mass),
        jnp.arange(log_N, dtype=jnp.int32),
    )
    return frontier_node_logits, frontier_node_idxs


def tree_attention_stochastic_weights_original(
    q: jnp.ndarray,
    k: jnp.ndarray,
    num_samples: int,
    rng: jax.Array,
    causal_frontier_logits: jnp.ndarray | None = None,
    causal_frontier_idxs: jnp.ndarray | None = None,
    q_idx: jnp.ndarray | None = None,
    dropout: float = 0.0,
    gumbel_scale: float = 1.0,
    max_logit: float = 1e3,
):
    """Original stochastic non-causal tree attention weights (matches PyTorch reference).

    Assumption: num_samples < N where N is number of leaves

    Args:
            q: [D]
            k: [K, D] where K = N-1 and N is number of leaves
            num_samples: S
            key: PRNGKey
            max_logit: clamp passed to weight computation

    Returns:
            outputs: [D]
            attn_weights: [S]
            sampled_idx: [S]
    """
    assert (
        causal_frontier_logits is None
        and causal_frontier_idxs is None
        and q_idx is None
    ) or (
        causal_frontier_logits is not None
        and causal_frontier_idxs is not None
        and q_idx is not None
    ), (
        "causal_frontier_logits and causal_frontier_idxs must be either both None or both not None"
    )
    K = k.shape[0]
    N = K + 1
    if K < 0:
        raise ValueError("invalid K")
    if K == 0:
        # Single leaf
        return jnp.array([1.0], dtype=q.dtype)

    log_N = int(K).bit_length()
    root_idx = 2 ** (log_N - 1) - 1

    cum_sample_log_probs = jnp.full((num_samples,), fill_value=-jnp.inf, dtype=q.dtype)
    cum_sample_log_probs = cum_sample_log_probs.at[0].set(0.0)
    cum_sample_gumbels = jnp.full((num_samples,), fill_value=0.0, dtype=q.dtype)
    current_sample_idxs = jnp.full((num_samples,), fill_value=root_idx, dtype=jnp.int32)
    # nodes in the tree that are disabled and will be skipped
    node_valid_mask = jnp.full((num_samples,), fill_value=True, dtype=jnp.bool_)

    def body(carry, top_down_lvl_idx):
        (
            cum_sample_log_probs,
            cum_sample_gumbels,
            current_sample_idxs,
            node_valid_mask,
            rng,
        ) = carry
        left_rng, right_rng, rng = jax.random.split(rng, 3)

        bottom_up_lvl_idx = log_N - 1 - top_down_lvl_idx

        def get_children_indices_internal_node(operands):
            bottom_up_lvl_idx, current_sample_idxs = operands
            parent_to_child_step = 1 << (bottom_up_lvl_idx - 1)
            left_child_key_idx = current_sample_idxs - parent_to_child_step
            right_child_key_idx = current_sample_idxs + parent_to_child_step
            return left_child_key_idx, right_child_key_idx

        def get_children_indices_leaf_node(operands):
            _, current_sample_idxs = operands
            left_child_key_idx = current_sample_idxs
            right_child_key_idx = current_sample_idxs + 1
            return left_child_key_idx, right_child_key_idx

        left_child_key_idx, right_child_key_idx = jax.lax.cond(
            top_down_lvl_idx < log_N - 1,
            get_children_indices_internal_node,
            get_children_indices_leaf_node,
            (bottom_up_lvl_idx, current_sample_idxs),
        )
        current_sample_idxs_clipped = jnp.clip(current_sample_idxs, 0, K - 1)
        logits = jnp.dot(q, k[current_sample_idxs_clipped].T)
        logits = jnp.clip(logits, -max_logit, max_logit)
        if causal_frontier_logits is not None and causal_frontier_idxs is not None:
            # if current_sample_idx is in causal frontier, use the causal frontier logits
            # otherwise compute logits from <q,K>
            layer_causal_key_idx = causal_frontier_idxs[bottom_up_lvl_idx]
            layer_causal_key_logit = causal_frontier_logits[bottom_up_lvl_idx]
            logits = jnp.where(
                current_sample_idxs == layer_causal_key_idx,
                layer_causal_key_logit,
                logits,
            )
            node_valid_mask = node_valid_mask.at[...].set(current_sample_idxs < q_idx)
        else:
            node_valid_mask = node_valid_mask.at[...].set(current_sample_idxs < K)
        left_log_probs = jnp.where(
            node_valid_mask,
            log_sigmoid(-logits) + cum_sample_log_probs,
            cum_sample_log_probs,
        )
        right_log_probs = jnp.where(
            node_valid_mask, log_sigmoid(logits) + cum_sample_log_probs, -jnp.inf
        )

        left_gumbel_samples = (
            gumbel(left_rng, (num_samples,), dtype=q.dtype) * gumbel_scale
            + left_log_probs
        )
        right_gumbel_samples = (
            gumbel(right_rng, (num_samples,), dtype=q.dtype) * gumbel_scale
            + right_log_probs
        )
        gumbel_level_max = jnp.max(
            jnp.concatenate(
                [jnp.asarray(left_gumbel_samples), jnp.asarray(right_gumbel_samples)],
                axis=0,
            ),
            axis=0,
        )

        gumbel_logsumexp_signs = jnp.array([1, 1, -1], dtype=q.dtype)
        gumbel_level_max_broadcasted = jnp.broadcast_to(
            gumbel_level_max, left_gumbel_samples.shape
        )
        gumbel_updates_left = jnp.stack(
            (-left_gumbel_samples, -cum_sample_gumbels, -gumbel_level_max_broadcasted),
            axis=0,
        )
        gumbel_updates_right = jnp.stack(
            (-right_gumbel_samples, -cum_sample_gumbels, -gumbel_level_max_broadcasted),
            axis=0,
        )

        left_gumbels = -jax.scipy.special.logsumexp(
            a=gumbel_updates_left, b=gumbel_logsumexp_signs[:, None], axis=0
        )
        right_gumbels = -jax.scipy.special.logsumexp(
            a=gumbel_updates_right, b=gumbel_logsumexp_signs[:, None], axis=0
        )

        cum_sample_gumbels, top_child_gumbel_idxs = jax.lax.top_k(
            jnp.concatenate(
                [jnp.asarray(left_gumbels), jnp.asarray(right_gumbels)], axis=0
            ),
            k=num_samples,
        )
        cum_child_log_probs = jnp.concatenate(
            [jnp.asarray(left_log_probs), jnp.asarray(right_log_probs)], axis=0
        )
        children_key_idxs = jnp.concatenate(
            [jnp.asarray(left_child_key_idx), jnp.asarray(right_child_key_idx)], axis=0
        )
        current_sample_idxs = children_key_idxs[top_child_gumbel_idxs]
        cum_sample_log_probs = cum_child_log_probs[top_child_gumbel_idxs]
        return (
            cum_sample_log_probs,
            cum_sample_gumbels,
            current_sample_idxs,
            node_valid_mask,
            rng,
        ), None

    (
        (
            cum_sample_log_probs,
            cum_sample_gumbels,
            current_sample_idxs,
            node_valid_mask,
            rng,
        ),
        _,
    ) = jax.lax.scan(
        body,
        (
            cum_sample_log_probs,
            cum_sample_gumbels,
            current_sample_idxs,
            node_valid_mask,
            rng,
        ),
        jnp.arange(0, log_N, dtype=jnp.int32),
    )
    attn_weights = jnp.exp(cum_sample_log_probs)
    return attn_weights, current_sample_idxs


@partial(
    jax.custom_vjp,
    nondiff_argnames=("num_samples", "dropout", "gumbel_scale", "max_logit"),
)
def tree_attention_stochastic_weights(
    q: jnp.ndarray,
    k: jnp.ndarray,
    num_samples: int,
    rng: jax.Array,
    causal_frontier_logits: jnp.ndarray | None = None,
    causal_frontier_idxs: jnp.ndarray | None = None,
    q_idx: jnp.ndarray | None = None,
    dropout: float = 0.0,
    gumbel_scale: float = 1.0,
    max_logit: float = 1e3,
):
    """Stochastic tree attention weights with custom VJP for efficient gradients.

    This version uses a custom backward pass that reconstructs paths from sampled
    indices, avoiding the need to store the full computation graph.
    """
    return tree_attention_stochastic_weights_original(
        q,
        k,
        num_samples,
        rng,
        causal_frontier_logits,
        causal_frontier_idxs,
        q_idx,
        dropout,
        gumbel_scale,
        max_logit,
    )


def tree_attention_stochastic_weights_fwd(
    q,
    k,
    num_samples,
    rng,
    causal_frontier_logits,
    causal_frontier_idxs,
    q_idx,
    dropout,
    gumbel_scale,
    max_logit,
):
    """Forward pass that saves minimal residuals for backward pass."""
    attn_weights, sampled_idxs = tree_attention_stochastic_weights_original(
        q,
        k,
        num_samples,
        rng,
        causal_frontier_logits,
        causal_frontier_idxs,
        q_idx,
        dropout,
        gumbel_scale,
        max_logit,
    )

    # Save only essential values for backward pass
    residuals = (
        q,
        k,
        causal_frontier_logits,
        causal_frontier_idxs,
        q_idx,
        sampled_idxs,
        attn_weights,
    )

    return (attn_weights, sampled_idxs), residuals


def tree_attention_stochastic_weights_bwd(
    num_samples, dropout, gumbel_scale, max_logit, residuals, grads
):
    """Backward pass that efficiently computes gradients by reconstructing paths."""
    (
        q,
        k,
        causal_frontier_logits,
        causal_frontier_idxs,
        q_idx,
        sampled_idxs,
        attn_weights,
    ) = residuals
    d_attn_weights, d_sampled_idxs = grads

    K = k.shape[0]
    N = K + 1

    if K == 0:
        # Single leaf case
        return (None, None, None, None, None, None)

    log_N = int(K).bit_length()

    # Streamed accumulation over samples and levels to avoid materializing [S, logN, D]
    if (causal_frontier_logits is not None) and (causal_frontier_idxs is not None):
        grad_q_acc = jnp.zeros_like(q)
        grad_k_acc = jnp.zeros_like(k)
        grad_causal_logits_acc = jnp.zeros((log_N,), dtype=q.dtype)

        def per_sample_step(carry, sample_triplet):
            grad_q_running, grad_k_running, grad_causal_logits_running = carry
            sample_idx, sample_weight, d_sample_weight = sample_triplet
            grad_contribution = d_sample_weight * sample_weight

            def per_level_step(level_carry, top_down_lvl_idx):
                grad_q_level, grad_k_level, grad_causal_logits_level = level_carry
                start = 1 << top_down_lvl_idx
                step = start << 1
                k_idx = (sample_idx // step) * step + start - 1
                safe_k_idx = jnp.clip(k_idx, 0, K - 1)
                bit = (sample_idx >> top_down_lvl_idx) & 1
                sign = jnp.where(bit == 0, -1.0, 1.0)
                layer_causal_idx = causal_frontier_idxs[top_down_lvl_idx]
                layer_causal_logit = causal_frontier_logits[top_down_lvl_idx]
                valid = k_idx < q_idx
                is_causal = (k_idx == layer_causal_idx) & valid
                non_causal_logit = jnp.dot(q, k[safe_k_idx])
                non_causal_logit = jnp.clip(non_causal_logit, -max_logit, max_logit)
                logit = jnp.where(is_causal, layer_causal_logit, non_causal_logit)
                grad_logit = sign * jax.nn.sigmoid(-sign * logit) * grad_contribution
                update_mask = (~is_causal) & valid
                weighted_update = jnp.where(update_mask, grad_logit, 0.0)
                grad_q_next = grad_q_level + weighted_update * k[safe_k_idx]
                grad_k_next = grad_k_level.at[safe_k_idx].add(weighted_update * q)
                grad_causal_logits_next = grad_causal_logits_level.at[
                    top_down_lvl_idx
                ].add(jnp.where(is_causal, grad_logit, 0.0))
                return (grad_q_next, grad_k_next, grad_causal_logits_next), None

            (grad_q_out, grad_k_out, grad_causal_logits_out), _ = jax.lax.scan(
                per_level_step,
                (grad_q_running, grad_k_running, grad_causal_logits_running),
                jnp.arange(log_N, dtype=jnp.int32),
            )
            return (grad_q_out, grad_k_out, grad_causal_logits_out), None

        (grad_q_acc, grad_k_acc, grad_causal_logits_acc), _ = jax.lax.scan(
            per_sample_step,
            (grad_q_acc, grad_k_acc, grad_causal_logits_acc),
            (sampled_idxs, attn_weights, d_attn_weights),
        )

        return (grad_q_acc, grad_k_acc, None, grad_causal_logits_acc, None, None)
    else:
        grad_q_acc = jnp.zeros_like(q)
        grad_k_acc = jnp.zeros_like(k)

        def per_sample_step_nc(carry, sample_triplet):
            grad_q_running, grad_k_running = carry
            sample_idx, sample_weight, d_sample_weight = sample_triplet
            grad_contribution = d_sample_weight * sample_weight

            def per_level_step_nc(level_carry, top_down_lvl_idx):
                grad_q_level, grad_k_level = level_carry
                start = 1 << top_down_lvl_idx
                step = start << 1
                k_idx = (sample_idx // step) * step + start - 1
                safe_k_idx = jnp.clip(k_idx, 0, K - 1)
                bit = (sample_idx >> top_down_lvl_idx) & 1
                sign = jnp.where(bit == 0, -1.0, 1.0)
                valid = k_idx < K
                logit = jnp.dot(q, k[safe_k_idx])
                logit = jnp.clip(logit, -max_logit, max_logit)
                grad_logit = sign * jax.nn.sigmoid(-sign * logit) * grad_contribution
                weighted_update = jnp.where(valid, grad_logit, 0.0)
                grad_q_next = grad_q_level + weighted_update * k[safe_k_idx]
                grad_k_next = grad_k_level.at[safe_k_idx].add(weighted_update * q)
                return (grad_q_next, grad_k_next), None

            (grad_q_out, grad_k_out), _ = jax.lax.scan(
                per_level_step_nc,
                (grad_q_running, grad_k_running),
                jnp.arange(log_N, dtype=jnp.int32),
            )
            return (grad_q_out, grad_k_out), None

        (grad_q_acc, grad_k_acc), _ = jax.lax.scan(
            per_sample_step_nc,
            (grad_q_acc, grad_k_acc),
            (sampled_idxs, attn_weights, d_attn_weights),
        )

        return (grad_q_acc, grad_k_acc, None, None, None, None)


# Register the VJP
tree_attention_stochastic_weights.defvjp(
    tree_attention_stochastic_weights_fwd, tree_attention_stochastic_weights_bwd
)


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "num_samples",
        "dropout",
        "gumbel_scale",
        "max_logit",
        "block_size",
    ),
)
def batched_tree_attention_stochastic_weights(
    q: jnp.ndarray,
    k: jnp.ndarray,
    rngs: jax.Array,
    causal_frontier_logits: jnp.ndarray,
    causal_frontier_idxs: jnp.ndarray,
    q_idxs: jnp.ndarray,
    num_samples: int,
    dropout: float = 0.0,
    gumbel_scale: float = 1.0,
    max_logit: float = 1e3,
    block_size: int = 128,
):
    """Batched wrapper around per-query stochastic weights with memory-efficient backward.

    Forward vmaps the per-query forward. Backward scans over queries in blocks, accumulating
    gradients to keys to avoid materializing a [L, K, D] buffer.
    """
    return batched_tree_attention_stochastic_weights_fwd(
        q,
        k,
        rngs,
        causal_frontier_logits,
        causal_frontier_idxs,
        q_idxs,
        num_samples,
        dropout,
        gumbel_scale,
        max_logit,
        block_size,
    )[0]


def batched_tree_attention_stochastic_weights_fwd(
    q,
    k,
    rngs,
    causal_frontier_logits,
    causal_frontier_idxs,
    q_idxs,
    num_samples,
    dropout,
    gumbel_scale,
    max_logit,
    block_size,
):
    """Forward pass: vmap per-query forward of tree_attention_stochastic_weights."""

    @partial(jax.vmap, in_axes=(0, None, 0, 0, 0, 0))
    def per_query_forward(_q, _k, _rng, _cf_logits, _cf_idxs, _q_idx):
        return tree_attention_stochastic_weights(
            q=_q,
            k=_k,
            num_samples=num_samples,
            rng=_rng,
            causal_frontier_logits=_cf_logits,
            causal_frontier_idxs=_cf_idxs,
            q_idx=_q_idx,
            dropout=dropout,
            gumbel_scale=gumbel_scale,
            max_logit=max_logit,
        )

    weights, sampled_idxs = per_query_forward(
        q, k, rngs, causal_frontier_logits, causal_frontier_idxs, q_idxs
    )
    residuals = (
        q,
        k,
        causal_frontier_logits,
        causal_frontier_idxs,
        q_idxs,
        sampled_idxs,
        weights,
    )
    return (weights, sampled_idxs), residuals


def batched_tree_attention_stochastic_weights_bwd(
    num_samples,
    dropout,
    gumbel_scale,
    max_logit,
    block_size,
    residuals,
    grads,
):
    """Backward pass scanning over queries in blocks to accumulate key-gradients.

    Avoids allocating [L, K, D] by processing queries sequentially in JAX and
    accumulating into a single [K, D] gradient buffer.
    """
    (
        q,
        k,
        causal_frontier_logits,
        causal_frontier_idxs,
        q_idxs,
        sampled_idxs,
        attn_weights,
    ) = residuals
    # Only the first component (weights) is differentiable; ignore gradient w.r.t. sampled indices
    d_attn_weights, _ = grads

    K = k.shape[0]
    log_N = int(K).bit_length()
    L = q.shape[0]

    # Number of blocks (Python int), avoid jnp.pad which requires concrete pad widths
    num_blocks = (L + block_size - 1) // block_size

    D = q.shape[1]
    grad_k_init = jnp.zeros_like(k)
    grad_q_out_init = jnp.zeros((L, D), dtype=q.dtype)
    grad_cf_out_init = jnp.zeros((L, log_N), dtype=q.dtype)

    def per_block(carry, block_idx):
        grad_k_running, grad_q_out, grad_cf_out = carry
        start = block_idx * block_size

        row_offsets = jnp.arange(block_size, dtype=jnp.int32)
        idx = start + row_offsets
        valid_mask = idx < L
        last_idx = jnp.maximum(L - 1, 0)
        idx_clipped = jnp.minimum(idx, last_idx)

        def gather_rows(x):
            return x[idx_clipped]

        q_block = gather_rows(q)
        cf_logits_block = gather_rows(causal_frontier_logits)
        cf_idxs_block = gather_rows(causal_frontier_idxs)
        q_idxs_block = gather_rows(q_idxs)
        sampled_idxs_block = gather_rows(sampled_idxs)
        attn_weights_block = gather_rows(attn_weights)
        d_attn_weights_block = gather_rows(d_attn_weights)

        # Zero-out padded rows to avoid any spurious compute/contributions
        q_block = jnp.where(valid_mask[:, None], q_block, 0.0)
        attn_weights_block = jnp.where(valid_mask[:, None], attn_weights_block, 0.0)
        d_attn_weights_block = jnp.where(valid_mask[:, None], d_attn_weights_block, 0.0)

        # Vectorized reconstruction of per-level indices and signs
        levels = jnp.arange(log_N, dtype=jnp.int32)
        level_offset = jnp.expand_dims(jnp.expand_dims(1 << levels, 0), 0)  # [1,1,L]
        level_step = level_offset << 1  # [1,1,L]
        # k indices visited by each sampled path at each level
        sample_idx_exp = jnp.expand_dims(sampled_idxs_block, -1)  # [B,S,1]
        k_idx = (
            (sample_idx_exp // level_step) * level_step + level_offset - 1
        )  # [B,S,L]
        safe_k_idx = jnp.clip(k_idx, 0, K - 1)
        bit = (sample_idx_exp >> levels) & 1
        sign = jnp.where(bit == 0, -1.0, 1.0).astype(q.dtype)
        # validity and causal masks
        valid = k_idx < q_idxs_block[:, None, None]
        is_causal = (k_idx == cf_idxs_block[:, None, :]) & valid

        # Gather K vectors and compute logits
        k_gathered = jnp.take(k, safe_k_idx, axis=0)  # [B,S,L,D]
        non_causal_logit = jnp.einsum("bd,bsld->bsl", q_block, k_gathered)
        non_causal_logit = jnp.clip(non_causal_logit, -max_logit, max_logit)
        cf_logits_b = jnp.expand_dims(cf_logits_block, 1)
        logit = jnp.where(is_causal, cf_logits_b, non_causal_logit)

        # Per-sample contribution shared across levels
        grad_contribution = jnp.expand_dims(
            jnp.asarray(d_attn_weights_block) * jnp.asarray(attn_weights_block), -1
        )
        grad_logit = sign * jax.nn.sigmoid(-sign * logit) * grad_contribution

        # Updates for q/k and causal frontier logits
        update_mask = (~is_causal) & valid
        weighted_update = jnp.where(update_mask, grad_logit, 0.0)

        # grad_q accumulation: sum over samples and levels
        grad_q_block = jnp.einsum("bsl,bsld->bd", weighted_update, k_gathered)
        # grad_cf accumulation per level: sum over samples
        grad_cf_block = jnp.sum(jnp.where(is_causal, grad_logit, 0.0), axis=1)

        # grad_k via scatter-add of flattened updates
        flat_indices = jnp.reshape(safe_k_idx, (-1,))
        q_block_expanded = jnp.reshape(q_block, (q_block.shape[0], 1, 1, D))
        wu = jnp.asarray(weighted_update)
        q_broadcast = jnp.broadcast_to(q_block_expanded, wu.shape + (D,))
        product_updates = wu[..., None] * q_broadcast
        flat_updates = jnp.reshape(product_updates, (-1, D))
        grad_k_block_sum = jnp.zeros_like(k).at[flat_indices].add(flat_updates)
        grad_k_block_out = grad_k_running + grad_k_block_sum

        # Scatter-add valid rows into outputs
        grad_q_out = grad_q_out.at[idx_clipped].add(grad_q_block)
        grad_cf_out = grad_cf_out.at[idx_clipped].add(grad_cf_block)
        return (grad_k_block_out, grad_q_out, grad_cf_out), None

    (grad_k_final, grad_q_full, grad_cf_full), _ = jax.lax.scan(
        per_block,
        (grad_k_init, grad_q_out_init, grad_cf_out_init),
        jnp.arange(num_blocks, dtype=jnp.int32),
    )

    return (grad_q_full, grad_k_final, None, grad_cf_full, None, None)


# Register the VJP for the batched wrapper
batched_tree_attention_stochastic_weights.defvjp(
    batched_tree_attention_stochastic_weights_fwd,
    batched_tree_attention_stochastic_weights_bwd,
)


def tree_attention_noncausal_stochastic(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    num_samples: int,
    rng: jax.Array,
    dropout: float | None = None,
    gumbel_scale: float = 1.0,
    max_logit: float = 1e3,
    return_weights: bool = False,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]] | jnp.ndarray:
    """Stochastic non-causal tree attention weights (matches PyTorch reference).

    Args:
            q: [D]
            k: [K, D] where K = N-1 and N is number of leaves
            v: [N, D]
            num_samples: S
            key: PRNGKey
            max_logit: clamp passed to weight computation
            return_weights: whether to return weights

    Returns:
            attn: [D]
            weights: [S] if return_weights is True, otherwise None
    """
    if dropout is not None:
        sample_rng, dropout_rng = jax.random.split(rng, 2)
    else:
        sample_rng, dropout_rng = rng, None

    weights, sampled_idxs = tree_attention_stochastic_weights(
        q, k, num_samples, sample_rng, gumbel_scale=gumbel_scale, max_logit=max_logit
    )
    if dropout is not None:
        dropout_layer = eqx.nn.Dropout(p=dropout, inference=False)
        weights = dropout_layer(weights, key=dropout_rng)
    attn = jnp.einsum("n, n d -> d", weights, v[sampled_idxs])
    if return_weights:
        return attn, (weights, sampled_idxs)
    else:
        return attn


def tree_attention_causal_stochastic(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    rng: jax.Array | jnp.ndarray,
    num_samples: int,
    dropout: float | None = None,
    gumbel_scale: float = 1.0,
    max_logit: float = 1e3,
    return_weights: bool = False,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]] | jnp.ndarray:
    """Stochastic causal tree attention weights (matches PyTorch reference).

    Args:
            q: [L, D]
            k: [K, D] where K = N-1 and N is number of leaves
            v: [N, D]
            num_samples: S
            key: PRNGKey array of shape [L]
            max_logit: clamp passed to weight computation
            return_weights: whether to return weights

    Returns:
            attn: [L, D]
            weights: [L, S] if return_weights is True, otherwise None
    """
    q_idxs = jnp.arange(q.shape[0], dtype=jnp.int32)

    @partial(jax.vmap, in_axes=(0, None, 0))
    def query_batched_query_causal_frontier_update(_q, _k, _q_idx):
        return query_causal_frontier_update(_q, _k, _q_idx, max_logit=max_logit)

    frontier_node_logits, frontier_node_idxs = (
        query_batched_query_causal_frontier_update(q, k, q_idxs)
    )

    if dropout is not None:
        # Split each per-query RNG into sampling and dropout RNGs
        splitted = jax.vmap(lambda key: jax.random.split(key, 2))(rng)
        sample_rngs = splitted[:, 0]
        dropout_rngs = splitted[:, 1]
    else:
        sample_rngs = rng
        dropout_rngs = None

    # Use memory-efficient batched custom-VJP with block-wise backward scan
    weights, sampled_idxs = batched_tree_attention_stochastic_weights(
        q=q,
        k=k,
        rngs=sample_rngs,
        causal_frontier_logits=frontier_node_logits,
        causal_frontier_idxs=frontier_node_idxs,
        q_idxs=q_idxs,
        num_samples=num_samples,
        dropout=dropout,
        gumbel_scale=gumbel_scale,
        max_logit=max_logit,
        block_size=256,
    )
    if dropout is not None:
        dropout_layer = eqx.nn.Dropout(p=dropout, inference=False)
        weights = jax.vmap(lambda x, key: dropout_layer(x, key=key))(
            weights, dropout_rngs
        )
    attn = jnp.einsum("l n, l n d -> l d", weights, v[sampled_idxs])
    if return_weights:
        return attn, (weights, sampled_idxs)
    else:
        return attn

