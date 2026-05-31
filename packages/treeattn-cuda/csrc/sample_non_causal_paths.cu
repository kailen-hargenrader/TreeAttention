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

__global__ void sample_non_causal_step_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    int64_t* __restrict__ current_nodes,
    float* __restrict__ accumulated_log_probs,
    uint8_t* __restrict__ packed_paths,
    const float* __restrict__ gumbels,
    int64_t k_len,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t block_queries,
    int64_t num_samples,
    int64_t path_bytes,
    int64_t depth,
    float max_logit) {
  const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = block_queries * batch * nheads * num_samples;
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
  const int64_t current_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * num_samples + sample_idx;
  const int64_t gumbel_offset = current_offset * 2;
  const int64_t packed_offset = current_offset * path_bytes;

  const int64_t node_idx = current_nodes[current_offset];
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

  const float left_log_prob = log_sigmoid_forward(dot);
  const float right_log_prob = log_sigmoid_forward(-dot);
  const float left_score = left_log_prob + gumbels[gumbel_offset];
  const float right_score = right_log_prob + gumbels[gumbel_offset + 1];
  const int64_t direction = right_score > left_score ? 1 : 0;
  const float chosen_log_prob = direction == 0 ? left_log_prob : right_log_prob;

  accumulated_log_probs[current_offset] += chosen_log_prob;
  packed_paths[packed_offset + (depth >> 3)] |= static_cast<uint8_t>(direction << (depth & 7));
  current_nodes[current_offset] = 2 * node_idx + 1 + direction;
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sample_non_causal_paths_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    int64_t num_samples,
    int64_t block_size,
    double max_logit) {
  c10::cuda::CUDAGuard device_guard(q.device());

  const auto num_queries = q.size(0);
  const auto batch = q.size(1);
  const auto nheads = q.size(2);
  const auto width = q.size(3);
  const auto k_len = k.size(0);
  const auto num_leaves = k_len + 1;
  const auto log_n = compute_log_n(num_leaves);
  const auto path_bytes = (log_n + 7) / 8;

  auto current_nodes = torch::zeros(
      {num_queries, batch, nheads, num_samples},
      q.options().dtype(torch::kInt64));
  auto accumulated_log_probs = torch::zeros(
      {num_queries, batch, nheads, num_samples},
      q.options().dtype(torch::kFloat32));
  auto packed_paths = torch::zeros(
      {num_queries, batch, nheads, num_samples, path_bytes},
      q.options().dtype(torch::kUInt8));

  constexpr int threads = 256;
  for (int64_t depth = 0; depth < log_n; ++depth) {
    auto gumbels = torch::empty(
      {num_queries, batch, nheads, num_samples, 2},
      q.options().dtype(torch::kFloat32));
    gumbels.exponential_();
    gumbels.log_();
    gumbels.neg_();
    for (int64_t start = 0; start < num_queries; start += block_size) {
      const int64_t block_queries = std::min(block_size, num_queries - start);
      auto q_block = q.narrow(0, start, block_queries);
      auto current_block = current_nodes.narrow(0, start, block_queries);
      auto log_prob_block = accumulated_log_probs.narrow(0, start, block_queries);
      auto packed_block = packed_paths.narrow(0, start, block_queries);
      auto gumbels_block = gumbels.narrow(0, start, block_queries);

      const int64_t total = block_queries * batch * nheads * num_samples;
      const int blocks = static_cast<int>((total + threads - 1) / threads);
      sample_non_causal_step_kernel<<<
          blocks,
          threads,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
          q_block.data_ptr<float>(),
          k.data_ptr<float>(),
          current_block.data_ptr<int64_t>(),
          log_prob_block.data_ptr<float>(),
          packed_block.data_ptr<uint8_t>(),
          gumbels_block.data_ptr<float>(),
          k_len,
          batch,
          nheads,
          width,
          block_queries,
          num_samples,
          path_bytes,
          depth,
          static_cast<float>(max_logit));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  auto sampled_indices = current_nodes - k_len;
  return std::make_tuple(sampled_indices, accumulated_log_probs, packed_paths);
}