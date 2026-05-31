#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float sigmoid_forward(float x) {
  if (x >= 0.0f) {
    const float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(x);
  return z / (1.0f + z);
}

__global__ void compute_grad_logit_non_causal_forward_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const int64_t* __restrict__ current_nodes,
    const uint8_t* __restrict__ packed_paths,
    const float* __restrict__ grad_log_probs,
    float* __restrict__ grad_logit,
    int64_t k_len,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples,
    int64_t path_bytes,
    int64_t depth,
    float max_logit) {
  const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_queries * batch * nheads * num_samples;
  if (linear_idx >= total) {
    return;
  }

  int64_t tmp = linear_idx;
  const int64_t sample_idx = tmp % num_samples;
  tmp /= num_samples;
  const int64_t head_idx = tmp % nheads;
  tmp /= nheads;
  const int64_t batch_idx = tmp % batch;
  const int64_t query_idx = tmp / batch;

  const int64_t current_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * num_samples + sample_idx;
  const int64_t node_idx = current_nodes[current_offset];
  const int64_t q_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * width;
  const int64_t k_offset = (((node_idx * batch) + batch_idx) * nheads + head_idx) * width;
  const int64_t packed_offset = current_offset * path_bytes;

  float dot = 0.0f;
  for (int64_t d = 0; d < width; ++d) {
    dot += q[q_offset + d] * k[k_offset + d];
  }
  if (dot > max_logit) {
    dot = max_logit;
  } else if (dot < -max_logit) {
    dot = -max_logit;
  }

  const uint8_t packed_byte = packed_paths[packed_offset + (depth >> 3)];
  const int64_t bit = static_cast<int64_t>((packed_byte >> (depth & 7)) & 1U);
  const float branch_sign = bit == 0 ? 1.0f : -1.0f;
  const float value = branch_sign * sigmoid_forward(-branch_sign * dot) * grad_log_probs[current_offset];
  grad_logit[current_offset] = value;
}

}  // namespace

torch::Tensor compute_grad_logit_non_causal_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& current_nodes,
    const torch::Tensor& packed_paths,
    const torch::Tensor& grad_log_probs,
    int64_t depth,
    double max_logit) {
  c10::cuda::CUDAGuard device_guard(q.device());

  const auto num_queries = q.size(0);
  const auto batch = q.size(1);
  const auto nheads = q.size(2);
  const auto width = q.size(3);
  const auto k_len = k.size(0);
  const auto num_samples = current_nodes.size(3);
  const auto path_bytes = packed_paths.size(4);

  auto grad_logit = torch::empty_like(grad_log_probs);

  constexpr int threads = 256;
  const int64_t total = num_queries * batch * nheads * num_samples;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  compute_grad_logit_non_causal_forward_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      current_nodes.data_ptr<int64_t>(),
      packed_paths.data_ptr<uint8_t>(),
      grad_log_probs.data_ptr<float>(),
      grad_logit.data_ptr<float>(),
      k_len,
      batch,
      nheads,
      width,
      num_queries,
      num_samples,
      path_bytes,
      depth,
      static_cast<float>(max_logit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_logit;
}