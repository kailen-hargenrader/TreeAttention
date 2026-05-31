#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <tuple>

namespace {

__device__ __forceinline__ float log_sigmoid_forward(float x) {
  if (x >= 0.0f) {
    return -log1pf(expf(-x));
  }
  return x - log1pf(expf(x));
}

__global__ void replay_non_causal_paths_forward_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const uint8_t* __restrict__ packed_paths,
    int64_t* __restrict__ sampled_indices,
    float* __restrict__ path_log_probs,
    int64_t k_len,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples,
    int64_t path_bytes,
    int64_t log_n,
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

  const int64_t q_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * width;
  const int64_t packed_offset = ((((query_idx * batch) + batch_idx) * nheads + head_idx) * num_samples + sample_idx) * path_bytes;

  int64_t node_idx = 0;
  float log_prob = 0.0f;
  for (int64_t depth = 0; depth < log_n; ++depth) {
    const int64_t byte_idx = depth >> 3;
    const int64_t bit_offset = depth & 7;
    const uint8_t packed_byte = packed_paths[packed_offset + byte_idx];
    const int64_t direction = static_cast<int64_t>((packed_byte >> bit_offset) & 1U);

    const int64_t k_offset = (((node_idx * batch) + batch_idx) * nheads + head_idx) * width;
    float dot = 0.0f;
    for (int64_t d = 0; d < width; ++d) {
      dot += q[q_offset + d] * k[k_offset + d];
    }

    if (dot > max_logit) {
      dot = max_logit;
    } else if (dot < -max_logit) {
      dot = -max_logit;
    }

    log_prob += direction == 0 ? log_sigmoid_forward(dot) : log_sigmoid_forward(-dot);
    node_idx = 2 * node_idx + 1 + direction;
  }

  sampled_indices[linear_idx] = node_idx - k_len;
  path_log_probs[linear_idx] = log_prob;
}

int64_t compute_log_n(int64_t num_leaves) {
  int64_t log_n = 0;
  while (num_leaves > 1) {
    ++log_n;
    num_leaves >>= 1;
  }
  return log_n;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> replay_non_causal_paths_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& packed_paths,
    double max_logit) {
  c10::cuda::CUDAGuard device_guard(q.device());

  const auto num_queries = q.size(0);
  const auto batch = q.size(1);
  const auto nheads = q.size(2);
  const auto width = q.size(3);
  const auto k_len = k.size(0);
  const auto num_samples = packed_paths.size(3);
  const auto path_bytes = packed_paths.size(4);
  const auto log_n = compute_log_n(k_len + 1);

  auto sampled_indices = torch::empty(
      {num_queries, batch, nheads, num_samples},
      q.options().dtype(torch::kInt64));
  auto path_log_probs = torch::empty(
      {num_queries, batch, nheads, num_samples},
      q.options().dtype(torch::kFloat32));

  constexpr int threads = 256;
  const int64_t total = num_queries * batch * nheads * num_samples;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  replay_non_causal_paths_forward_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      packed_paths.data_ptr<uint8_t>(),
      sampled_indices.data_ptr<int64_t>(),
      path_log_probs.data_ptr<float>(),
      k_len,
      batch,
      nheads,
      width,
      num_queries,
      num_samples,
      path_bytes,
      log_n,
      static_cast<float>(max_logit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(sampled_indices, path_log_probs);
}