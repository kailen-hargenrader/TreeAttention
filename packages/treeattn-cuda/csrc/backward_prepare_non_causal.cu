#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <tuple>

namespace {

__device__ __forceinline__ float log_sigmoid_forward(float x) {
  if (x >= 0.0f) {
    return -log1pf(expf(-x));
  }
  return x - log1pf(expf(x));
}

int64_t compute_log_n(int64_t num_leaves) {
  int64_t log_n = 0;
  while (num_leaves > 1) {
    ++log_n;
    num_leaves >>= 1;
  }
  return log_n;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return __shfl_sync(0xffffffffU, value, 0);
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return __shfl_sync(0xffffffffU, value, 0);
}

template <typename scalar_t, typename acc_t>
__global__ void prepare_non_causal_backward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const uint8_t* __restrict__ packed_paths,
    int32_t* __restrict__ sampled_indices,
    float* __restrict__ attn_weights,
    int64_t k_len,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples,
    int64_t path_bytes,
    int64_t log_n,
    float max_logit) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int64_t qbh_idx = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
  const int64_t total_qbh = num_queries * batch * nheads;
  if (qbh_idx >= total_qbh) {
    return;
  }

  const bool active = lane < num_samples;

  int64_t tmp = qbh_idx;
  const int64_t head_idx = tmp % nheads;
  tmp /= nheads;
  const int64_t batch_idx = tmp % batch;
  const int64_t query_idx = tmp / batch;

  const int64_t q_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * width;

  float log_prob = -INFINITY;
  int32_t leaf_idx = 0;
  if (active) {
    const int64_t sample_idx = lane;
    const int64_t packed_offset = ((((query_idx * batch) + batch_idx) * nheads + head_idx) * num_samples + sample_idx) * path_bytes;

    int64_t node_idx = 0;
    log_prob = 0.0f;
    for (int64_t depth = 0; depth < log_n; ++depth) {
      const uint8_t packed_byte = packed_paths[packed_offset + (depth >> 3)];
      const int64_t direction = static_cast<int64_t>((packed_byte >> (depth & 7)) & 1U);
      const int64_t k_offset = (((node_idx * batch) + batch_idx) * nheads + head_idx) * width;

      acc_t dot = static_cast<acc_t>(0);
      for (int64_t d = 0; d < width; ++d) {
        dot += static_cast<acc_t>(q[q_offset + d]) * static_cast<acc_t>(k[k_offset + d]);
      }

      const acc_t max_logit_acc = static_cast<acc_t>(max_logit);
      if (dot > max_logit_acc) {
        dot = max_logit_acc;
      } else if (dot < -max_logit_acc) {
        dot = -max_logit_acc;
      }

      log_prob += direction == 0 ? log_sigmoid_forward(static_cast<float>(dot))
                                 : log_sigmoid_forward(static_cast<float>(-dot));
      node_idx = 2 * node_idx + 1 + direction;
    }

    leaf_idx = static_cast<int32_t>(node_idx - k_len);
  }

  const float max_log_prob = warp_reduce_max(log_prob);

  float attn_weight = 0.0f;
  if (active) {
    attn_weight = expf(log_prob - max_log_prob);
  }

  const float sum_exp = warp_reduce_sum(attn_weight);

  if (active) {
    attn_weight /= sum_exp;
    const int64_t sample_offset = qbh_idx * num_samples + lane;
    sampled_indices[sample_offset] = leaf_idx;
    attn_weights[sample_offset] = attn_weight;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor>
prepare_non_causal_backward_cuda(
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
      q.options().dtype(torch::kInt32));
  auto attn_weights = torch::empty(
      {num_queries, batch, nheads, num_samples},
      q.options().dtype(torch::kFloat32));

  constexpr int threads = 128;
  constexpr int warps_per_block = threads / 32;
  const int64_t total_qbh = num_queries * batch * nheads;
  const int blocks = static_cast<int>((total_qbh + warps_per_block - 1) / warps_per_block);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "prepare_non_causal_backward_cuda",
      [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        prepare_non_causal_backward_kernel<scalar_t, acc_t><<<
            blocks,
            threads,
            0,
            at::cuda::getDefaultCUDAStream()>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            packed_paths.data_ptr<uint8_t>(),
            sampled_indices.data_ptr<int32_t>(),
            attn_weights.data_ptr<float>(),
            k_len,
            batch,
            nheads,
            width,
            num_queries,
            num_samples,
            path_bytes,
            log_n,
            static_cast<float>(max_logit));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(sampled_indices, attn_weights);
}