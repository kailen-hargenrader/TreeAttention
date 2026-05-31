#include <ATen/AccumulateType.h>
#include <ATen/cuda/Atomic.cuh>
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

int64_t compute_log_n_host(int64_t num_leaves) {
  int64_t log_n = 0;
  while (num_leaves > 1) {
    ++log_n;
    num_leaves >>= 1;
  }
  return log_n;
}
template <typename scalar_t, typename acc_t>
__global__ void accumulate_qk_non_causal_all_depths_inplace_kernel(
  const scalar_t* __restrict__ q,
  const scalar_t* __restrict__ k,
    const uint8_t* __restrict__ packed_paths,
    const float* __restrict__ grad_log_probs,
    float* __restrict__ grad_q_out,
    float* __restrict__ grad_k_out,
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

  const int64_t sample_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * num_samples + sample_idx;
  const int64_t q_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * width;
  const int64_t packed_offset = sample_offset * path_bytes;

  int64_t node_idx = 0;
  const float sample_grad_log_probs = grad_log_probs[sample_offset];
  for (int64_t depth = 0; depth < log_n; ++depth) {
    const uint8_t packed_byte = packed_paths[packed_offset + (depth >> 3)];
    const int64_t bit = static_cast<int64_t>((packed_byte >> (depth & 7)) & 1U);
    const float branch_sign = bit == 0 ? 1.0f : -1.0f;
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

    const float grad_logit = branch_sign * sigmoid_forward(-branch_sign * static_cast<float>(dot)) * sample_grad_log_probs;
    for (int64_t d = 0; d < width; ++d) {
      atomicAdd(grad_q_out + q_offset + d, grad_logit * static_cast<float>(k[k_offset + d]));
      atomicAdd(grad_k_out + k_offset + d, grad_logit * static_cast<float>(q[q_offset + d]));
    }

    node_idx = 2 * node_idx + 1 + bit;
  }
}

}  // namespace

template <typename scalar_t, typename acc_t>
void launch_accumulate_qk_non_causal_all_depths_inplace_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& packed_paths,
    const torch::Tensor& grad_log_probs,
    const torch::Tensor& grad_q_out,
    const torch::Tensor& grad_k_out,
    int64_t k_len,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples,
    int64_t path_bytes,
    int64_t log_n,
    float max_logit,
    int blocks,
    int threads) {
  accumulate_qk_non_causal_all_depths_inplace_kernel<scalar_t, acc_t><<<
      blocks,
      threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
      q.data_ptr<scalar_t>(),
      k.data_ptr<scalar_t>(),
      packed_paths.data_ptr<uint8_t>(),
      grad_log_probs.data_ptr<float>(),
      grad_q_out.data_ptr<float>(),
      grad_k_out.data_ptr<float>(),
      k_len,
      batch,
      nheads,
      width,
      num_queries,
      num_samples,
      path_bytes,
      log_n,
      max_logit);
}

void accumulate_qk_non_causal_all_depths_inplace_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& packed_paths,
    const torch::Tensor& grad_log_probs,
    double max_logit,
    const torch::Tensor& grad_q_out,
    const torch::Tensor& grad_k_out) {
  c10::cuda::CUDAGuard device_guard(q.device());

  const auto num_queries = q.size(0);
  const auto batch = q.size(1);
  const auto nheads = q.size(2);
  const auto width = q.size(3);
  const auto k_len = k.size(0);
  const auto num_samples = packed_paths.size(3);
  const auto path_bytes = packed_paths.size(4);
  const auto log_n = compute_log_n_host(k_len + 1);

  constexpr int threads = 256;
  const int64_t total = num_queries * batch * nheads * num_samples;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "accumulate_qk_non_causal_all_depths_inplace_cuda",
      [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        launch_accumulate_qk_non_causal_all_depths_inplace_kernel<scalar_t, acc_t>(
            q,
            k,
            packed_paths,
            grad_log_probs,
            grad_q_out,
            grad_k_out,
            k_len,
            batch,
            nheads,
            width,
            num_queries,
            num_samples,
            path_bytes,
            log_n,
            static_cast<float>(max_logit),
            blocks,
            threads);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}