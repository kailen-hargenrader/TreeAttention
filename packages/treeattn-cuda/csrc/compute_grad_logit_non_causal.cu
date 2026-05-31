#include <ATen/AccumulateType.h>
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

template <typename scalar_t, typename acc_t, typename index_t>
__global__ void compute_grad_logit_non_causal_forward_kernel(
  const scalar_t* __restrict__ q,
  const scalar_t* __restrict__ k,
  const index_t* __restrict__ current_nodes,
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
  const int64_t node_idx = static_cast<int64_t>(current_nodes[current_offset]);
  const int64_t q_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * width;
  const int64_t k_offset = (((node_idx * batch) + batch_idx) * nheads + head_idx) * width;
  const int64_t packed_offset = current_offset * path_bytes;

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

  const uint8_t packed_byte = packed_paths[packed_offset + (depth >> 3)];
  const int64_t bit = static_cast<int64_t>((packed_byte >> (depth & 7)) & 1U);
  const float branch_sign = bit == 0 ? 1.0f : -1.0f;
  const float value = branch_sign * sigmoid_forward(-branch_sign * static_cast<float>(dot)) * grad_log_probs[current_offset];
  grad_logit[current_offset] = value;
}

}  // namespace

template <typename scalar_t, typename acc_t>
void launch_compute_grad_logit_non_causal_forward_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& current_nodes,
    const torch::Tensor& packed_paths,
    const torch::Tensor& grad_log_probs,
    torch::Tensor& grad_logit,
    int64_t k_len,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples,
    int64_t path_bytes,
    int64_t depth,
    float max_logit,
    int blocks,
    int threads) {
  if (current_nodes.scalar_type() == torch::kInt32) {
    compute_grad_logit_non_causal_forward_kernel<scalar_t, acc_t, int32_t><<<
        blocks,
        threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
        q.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        current_nodes.data_ptr<int32_t>(),
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
        max_logit);
    return;
  }

  compute_grad_logit_non_causal_forward_kernel<scalar_t, acc_t, int64_t><<<
      blocks,
      threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
      q.data_ptr<scalar_t>(),
      k.data_ptr<scalar_t>(),
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
      max_logit);
}

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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "compute_grad_logit_non_causal_forward_cuda",
      [&] {
        using acc_t = at::acc_type<scalar_t, true>;
      launch_compute_grad_logit_non_causal_forward_kernel<scalar_t, acc_t>(
        q,
        k,
        current_nodes,
        packed_paths,
        grad_log_probs,
        grad_logit,
        k_len,
        batch,
        nheads,
        width,
        num_queries,
        num_samples,
        path_bytes,
        depth,
        static_cast<float>(max_logit),
        blocks,
        threads);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_logit;
}