#include <torch/extension.h>

#include <tuple>

namespace {

bool is_supported_treeattn_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
         dtype == torch::kBFloat16;
}

bool is_supported_treeattn_index_dtype(torch::ScalarType dtype) {
  return dtype == torch::kInt32 || dtype == torch::kInt64;
}

}  // namespace

torch::Tensor weighted_value_sum_forward_cuda(
    const torch::Tensor& v,
    const torch::Tensor& sampled_indices,
    const torch::Tensor& attn_weights);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sample_non_causal_paths_forward_cuda(
  const torch::Tensor& q,
  const torch::Tensor& k,
  int64_t num_samples,
  int64_t block_size,
  double max_logit);

std::tuple<torch::Tensor, torch::Tensor> replay_non_causal_paths_forward_cuda(
  const torch::Tensor& q,
  const torch::Tensor& k,
  const torch::Tensor& packed_paths,
  double max_logit);

torch::Tensor scatter_weighted_grad_v_forward_cuda(
  const torch::Tensor& grad_output,
  const torch::Tensor& sampled_indices,
  const torch::Tensor& attn_weights,
  int64_t num_leaves);

torch::Tensor compute_grad_logit_non_causal_forward_cuda(
  const torch::Tensor& q,
  const torch::Tensor& k,
  const torch::Tensor& current_nodes,
  const torch::Tensor& packed_paths,
  const torch::Tensor& grad_log_probs,
  int64_t depth,
  double max_logit);

void accumulate_qk_non_causal_inplace_cuda(
  const torch::Tensor& q,
  const torch::Tensor& k,
  const torch::Tensor& current_nodes,
  const torch::Tensor& packed_paths,
  const torch::Tensor& grad_log_probs,
  int64_t depth,
  double max_logit,
  const torch::Tensor& grad_q_out,
  const torch::Tensor& grad_k_out);

void accumulate_qk_non_causal_all_depths_inplace_cuda(
  const torch::Tensor& q,
  const torch::Tensor& k,
  const torch::Tensor& packed_paths,
  const torch::Tensor& grad_log_probs,
  double max_logit,
  const torch::Tensor& grad_q_out,
  const torch::Tensor& grad_k_out);

torch::Tensor weighted_value_sum_forward(
    const torch::Tensor& v,
    const torch::Tensor& sampled_indices,
    const torch::Tensor& attn_weights) {
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(sampled_indices.is_cuda(), "sampled_indices must be a CUDA tensor");
  TORCH_CHECK(attn_weights.is_cuda(), "attn_weights must be a CUDA tensor");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(sampled_indices.is_contiguous(), "sampled_indices must be contiguous");
  TORCH_CHECK(attn_weights.is_contiguous(), "attn_weights must be contiguous");
  TORCH_CHECK(v.dim() == 4, "v must have shape (N, B, H, D)");
  TORCH_CHECK(sampled_indices.dim() == 4, "sampled_indices must have shape (L, B, H, S)");
  TORCH_CHECK(attn_weights.dim() == 4, "attn_weights must have shape (L, B, H, S)");
  TORCH_CHECK(
      is_supported_treeattn_index_dtype(sampled_indices.scalar_type()),
      "sampled_indices must be int32 or int64");
  TORCH_CHECK(attn_weights.scalar_type() == torch::kFloat32, "attn_weights must be float32");
  TORCH_CHECK(sampled_indices.sizes() == attn_weights.sizes(), "sampled_indices and attn_weights must have the same shape");
  TORCH_CHECK(v.size(1) == sampled_indices.size(1), "batch dimensions must match");
  TORCH_CHECK(v.size(2) == sampled_indices.size(2), "head dimensions must match");

  return weighted_value_sum_forward_cuda(v, sampled_indices, attn_weights);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sample_non_causal_paths_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    int64_t num_samples,
    int64_t block_size,
    double max_logit) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(q.dim() == 4, "q must have shape (L, B, H, D)");
  TORCH_CHECK(k.dim() == 4, "k must have shape (K, B, H, D)");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
  TORCH_CHECK(
      is_supported_treeattn_dtype(q.scalar_type()),
      "q and k must be float32, float16, or bfloat16");
  TORCH_CHECK(k.size(0) > 0, "sample_non_causal_paths_forward expects K > 0");
  TORCH_CHECK(num_samples > 0, "num_samples must be > 0");
  TORCH_CHECK(block_size > 0, "block_size must be > 0");
  TORCH_CHECK(q.size(1) == k.size(1), "batch dimensions must match");
  TORCH_CHECK(q.size(2) == k.size(2), "head dimensions must match");

  return sample_non_causal_paths_forward_cuda(
      q, k, num_samples, block_size, max_logit);
}

std::tuple<torch::Tensor, torch::Tensor> replay_non_causal_paths_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& packed_paths,
    double max_logit) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(packed_paths.is_cuda(), "packed_paths must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(packed_paths.is_contiguous(), "packed_paths must be contiguous");
  TORCH_CHECK(q.dim() == 4, "q must have shape (L, B, H, D)");
  TORCH_CHECK(k.dim() == 4, "k must have shape (K, B, H, D)");
  TORCH_CHECK(
      packed_paths.dim() == 5,
      "packed_paths must have shape (L, B, H, S, P)");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
    TORCH_CHECK(
      is_supported_treeattn_dtype(q.scalar_type()),
      "q and k must be float32, float16, or bfloat16");
  TORCH_CHECK(
      packed_paths.scalar_type() == torch::kUInt8,
      "packed_paths must be uint8");
  TORCH_CHECK(k.size(0) > 0, "replay_non_causal_paths_forward expects K > 0");
  TORCH_CHECK(q.size(0) == packed_paths.size(0), "query length must match packed paths");
  TORCH_CHECK(q.size(1) == k.size(1), "batch dimensions must match");
  TORCH_CHECK(q.size(2) == k.size(2), "head dimensions must match");
  TORCH_CHECK(
      q.size(1) == packed_paths.size(1) && q.size(2) == packed_paths.size(2),
      "packed_paths batch/head dimensions must match q");

  return replay_non_causal_paths_forward_cuda(q, k, packed_paths, max_logit);
}

torch::Tensor scatter_weighted_grad_v_forward(
    const torch::Tensor& grad_output,
    const torch::Tensor& sampled_indices,
    const torch::Tensor& attn_weights,
    int64_t num_leaves) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(sampled_indices.is_cuda(), "sampled_indices must be a CUDA tensor");
  TORCH_CHECK(attn_weights.is_cuda(), "attn_weights must be a CUDA tensor");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
  TORCH_CHECK(sampled_indices.is_contiguous(), "sampled_indices must be contiguous");
  TORCH_CHECK(attn_weights.is_contiguous(), "attn_weights must be contiguous");
  TORCH_CHECK(grad_output.dim() == 4, "grad_output must have shape (L, B, H, D)");
  TORCH_CHECK(sampled_indices.dim() == 4, "sampled_indices must have shape (L, B, H, S)");
  TORCH_CHECK(attn_weights.dim() == 4, "attn_weights must have shape (L, B, H, S)");
  TORCH_CHECK(
      is_supported_treeattn_dtype(grad_output.scalar_type()),
      "grad_output must be float32, float16, or bfloat16");
  TORCH_CHECK(attn_weights.scalar_type() == torch::kFloat32, "attn_weights must be float32");
  TORCH_CHECK(
      is_supported_treeattn_index_dtype(sampled_indices.scalar_type()),
      "sampled_indices must be int32 or int64");
  TORCH_CHECK(sampled_indices.sizes() == attn_weights.sizes(), "sampled_indices and attn_weights must have the same shape");
  TORCH_CHECK(grad_output.size(0) == sampled_indices.size(0), "query length must match");
  TORCH_CHECK(grad_output.size(1) == sampled_indices.size(1), "batch dimensions must match");
  TORCH_CHECK(grad_output.size(2) == sampled_indices.size(2), "head dimensions must match");
  TORCH_CHECK(num_leaves > 0, "num_leaves must be > 0");

  return scatter_weighted_grad_v_forward_cuda(
      grad_output, sampled_indices, attn_weights, num_leaves);
}

torch::Tensor compute_grad_logit_non_causal_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& current_nodes,
    const torch::Tensor& packed_paths,
    const torch::Tensor& grad_log_probs,
    int64_t depth,
    double max_logit) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(current_nodes.is_cuda(), "current_nodes must be a CUDA tensor");
  TORCH_CHECK(packed_paths.is_cuda(), "packed_paths must be a CUDA tensor");
  TORCH_CHECK(grad_log_probs.is_cuda(), "grad_log_probs must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(current_nodes.is_contiguous(), "current_nodes must be contiguous");
  TORCH_CHECK(packed_paths.is_contiguous(), "packed_paths must be contiguous");
  TORCH_CHECK(grad_log_probs.is_contiguous(), "grad_log_probs must be contiguous");
  TORCH_CHECK(q.dim() == 4, "q must have shape (L, B, H, D)");
  TORCH_CHECK(k.dim() == 4, "k must have shape (K, B, H, D)");
  TORCH_CHECK(current_nodes.dim() == 4, "current_nodes must have shape (L, B, H, S)");
  TORCH_CHECK(packed_paths.dim() == 5, "packed_paths must have shape (L, B, H, S, P)");
  TORCH_CHECK(grad_log_probs.dim() == 4, "grad_log_probs must have shape (L, B, H, S)");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
  TORCH_CHECK(
      is_supported_treeattn_dtype(q.scalar_type()),
      "q and k must be float32, float16, or bfloat16");
  TORCH_CHECK(
      is_supported_treeattn_index_dtype(current_nodes.scalar_type()),
      "current_nodes must be int32 or int64");
  TORCH_CHECK(packed_paths.scalar_type() == torch::kUInt8, "packed_paths must be uint8");
  TORCH_CHECK(grad_log_probs.scalar_type() == torch::kFloat32, "grad_log_probs must be float32");
  TORCH_CHECK(depth >= 0, "depth must be >= 0");
  TORCH_CHECK(q.size(0) == current_nodes.size(0), "query length must match current_nodes");
  TORCH_CHECK(q.size(0) == packed_paths.size(0), "query length must match packed_paths");
  TORCH_CHECK(q.size(0) == grad_log_probs.size(0), "query length must match grad_log_probs");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(2) == k.size(2), "q and k batch/head dimensions must match");
  TORCH_CHECK(q.size(1) == current_nodes.size(1) && q.size(2) == current_nodes.size(2), "current_nodes batch/head dimensions must match q");
  TORCH_CHECK(q.size(1) == packed_paths.size(1) && q.size(2) == packed_paths.size(2), "packed_paths batch/head dimensions must match q");
  TORCH_CHECK(q.size(1) == grad_log_probs.size(1) && q.size(2) == grad_log_probs.size(2), "grad_log_probs batch/head dimensions must match q");
  TORCH_CHECK(current_nodes.sizes() == grad_log_probs.sizes(), "current_nodes and grad_log_probs must have the same shape");

  return compute_grad_logit_non_causal_forward_cuda(
      q, k, current_nodes, packed_paths, grad_log_probs, depth, max_logit);
}

void accumulate_qk_non_causal_inplace(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& current_nodes,
    const torch::Tensor& packed_paths,
    const torch::Tensor& grad_log_probs,
    int64_t depth,
    double max_logit,
    const torch::Tensor& grad_q_out,
    const torch::Tensor& grad_k_out) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(current_nodes.is_cuda(), "current_nodes must be a CUDA tensor");
  TORCH_CHECK(packed_paths.is_cuda(), "packed_paths must be a CUDA tensor");
  TORCH_CHECK(grad_log_probs.is_cuda(), "grad_log_probs must be a CUDA tensor");
  TORCH_CHECK(grad_q_out.is_cuda(), "grad_q_out must be a CUDA tensor");
  TORCH_CHECK(grad_k_out.is_cuda(), "grad_k_out must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(current_nodes.is_contiguous(), "current_nodes must be contiguous");
  TORCH_CHECK(packed_paths.is_contiguous(), "packed_paths must be contiguous");
  TORCH_CHECK(grad_log_probs.is_contiguous(), "grad_log_probs must be contiguous");
  TORCH_CHECK(q.dim() == 4, "q must have shape (L, B, H, D)");
  TORCH_CHECK(k.dim() == 4, "k must have shape (K, B, H, D)");
  TORCH_CHECK(current_nodes.dim() == 4, "current_nodes must have shape (L, B, H, S)");
  TORCH_CHECK(packed_paths.dim() == 5, "packed_paths must have shape (L, B, H, S, P)");
  TORCH_CHECK(grad_log_probs.dim() == 4, "grad_log_probs must have shape (L, B, H, S)");
  TORCH_CHECK(grad_q_out.dim() == 4, "grad_q_out must have shape (L, B, H, D)");
  TORCH_CHECK(grad_k_out.dim() == 4, "grad_k_out must have shape (K, B, H, D)");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
  TORCH_CHECK(
      is_supported_treeattn_dtype(q.scalar_type()),
      "q and k must be float32, float16, or bfloat16");
  TORCH_CHECK(
      is_supported_treeattn_index_dtype(current_nodes.scalar_type()),
      "current_nodes must be int32 or int64");
  TORCH_CHECK(packed_paths.scalar_type() == torch::kUInt8, "packed_paths must be uint8");
  TORCH_CHECK(grad_log_probs.scalar_type() == torch::kFloat32, "grad_log_probs must be float32");
  TORCH_CHECK(grad_q_out.scalar_type() == torch::kFloat32, "grad_q_out must be float32");
  TORCH_CHECK(grad_k_out.scalar_type() == torch::kFloat32, "grad_k_out must be float32");
  TORCH_CHECK(grad_k_out.scalar_type() == torch::kFloat32, "grad_k_out must be float32");
  TORCH_CHECK(k.size(0) == grad_k_out.size(0), "key length must match grad_k_out");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(2) == k.size(2), "q and k batch/head dimensions must match");
  TORCH_CHECK(q.size(1) == grad_q_out.size(1) && q.size(2) == grad_q_out.size(2), "grad_q_out batch/head dimensions must match q");
  TORCH_CHECK(k.size(1) == grad_k_out.size(1) && k.size(2) == grad_k_out.size(2), "grad_k_out batch/head dimensions must match k");

  accumulate_qk_non_causal_inplace_cuda(
      q,
      k,
      current_nodes,
      packed_paths,
      grad_log_probs,
      depth,
      max_logit,
      grad_q_out,
      grad_k_out);
}

void accumulate_qk_non_causal_all_depths_inplace(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& packed_paths,
    const torch::Tensor& grad_log_probs,
    double max_logit,
    const torch::Tensor& grad_q_out,
    const torch::Tensor& grad_k_out) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(packed_paths.is_cuda(), "packed_paths must be a CUDA tensor");
  TORCH_CHECK(grad_log_probs.is_cuda(), "grad_log_probs must be a CUDA tensor");
  TORCH_CHECK(grad_q_out.is_cuda(), "grad_q_out must be a CUDA tensor");
  TORCH_CHECK(grad_k_out.is_cuda(), "grad_k_out must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(packed_paths.is_contiguous(), "packed_paths must be contiguous");
  TORCH_CHECK(grad_log_probs.is_contiguous(), "grad_log_probs must be contiguous");
  TORCH_CHECK(q.dim() == 4, "q must have shape (L, B, H, D)");
  TORCH_CHECK(k.dim() == 4, "k must have shape (K, B, H, D)");
  TORCH_CHECK(packed_paths.dim() == 5, "packed_paths must have shape (L, B, H, S, P)");
  TORCH_CHECK(grad_log_probs.dim() == 4, "grad_log_probs must have shape (L, B, H, S)");
  TORCH_CHECK(grad_q_out.dim() == 4, "grad_q_out must have shape (L, B, H, D)");
  TORCH_CHECK(grad_k_out.dim() == 4, "grad_k_out must have shape (K, B, H, D)");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
  TORCH_CHECK(
      is_supported_treeattn_dtype(q.scalar_type()),
      "q and k must be float32, float16, or bfloat16");
  TORCH_CHECK(packed_paths.scalar_type() == torch::kUInt8, "packed_paths must be uint8");
  TORCH_CHECK(grad_log_probs.scalar_type() == torch::kFloat32, "grad_log_probs must be float32");
  TORCH_CHECK(grad_q_out.scalar_type() == torch::kFloat32, "grad_q_out must be float32");
    TORCH_CHECK(
      grad_k_out.scalar_type() == torch::kFloat32 || grad_k_out.scalar_type() == q.scalar_type(),
      "grad_k_out must be float32 or match q/k dtype");
  TORCH_CHECK(q.size(0) == packed_paths.size(0), "query length must match packed_paths");
  TORCH_CHECK(q.size(0) == grad_log_probs.size(0), "query length must match grad_log_probs");
  TORCH_CHECK(q.size(0) == grad_q_out.size(0), "query length must match grad_q_out");
  TORCH_CHECK(k.size(0) == grad_k_out.size(0), "key length must match grad_k_out");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(2) == k.size(2), "q and k batch/head dimensions must match");
  TORCH_CHECK(q.size(1) == packed_paths.size(1) && q.size(2) == packed_paths.size(2), "packed_paths batch/head dimensions must match q");
  TORCH_CHECK(q.size(1) == grad_log_probs.size(1) && q.size(2) == grad_log_probs.size(2), "grad_log_probs batch/head dimensions must match q");
  TORCH_CHECK(q.size(1) == grad_q_out.size(1) && q.size(2) == grad_q_out.size(2), "grad_q_out batch/head dimensions must match q");
  TORCH_CHECK(k.size(1) == grad_k_out.size(1) && k.size(2) == grad_k_out.size(2), "grad_k_out batch/head dimensions must match k");

  accumulate_qk_non_causal_all_depths_inplace_cuda(
      q,
      k,
      packed_paths,
      grad_log_probs,
      max_logit,
      grad_q_out,
      grad_k_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("accumulate_qk_non_causal_all_depths_inplace", &accumulate_qk_non_causal_all_depths_inplace, "Accumulate full non-causal q/k backward contributions in place (CUDA)");
  m.def("accumulate_qk_non_causal_inplace", &accumulate_qk_non_causal_inplace, "Accumulate non-causal q/k backward contributions in place (CUDA)");
  m.def("compute_grad_logit_non_causal_forward", &compute_grad_logit_non_causal_forward, "Compute non-causal grad_logit for one backward depth (CUDA)");
  m.def("sample_non_causal_paths_forward", &sample_non_causal_paths_forward, "Sample non-causal tree paths forward (CUDA)");
  m.def("scatter_weighted_grad_v_forward", &scatter_weighted_grad_v_forward, "Scatter weighted grad_v updates (CUDA)");
  m.def("weighted_value_sum_forward", &weighted_value_sum_forward, "Weighted tree value sum forward (CUDA)");
  m.def("replay_non_causal_paths_forward", &replay_non_causal_paths_forward, "Replay packed non-causal tree paths (CUDA)");
}