#include <ATen/AccumulateType.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t, typename out_t, typename index_t>
__global__ void scatter_weighted_grad_v_forward_kernel(
  const scalar_t* __restrict__ grad_output,
  const index_t* __restrict__ sampled_indices,
    const float* __restrict__ attn_weights,
    out_t* __restrict__ grad_v,
    int64_t num_leaves,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples) {
  const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_queries * batch * nheads * num_samples * width;
  if (linear_idx >= total) {
    return;
  }

  int64_t tmp = linear_idx;
  const int64_t d = tmp % width;
  tmp /= width;
  const int64_t sample_idx = tmp % num_samples;
  tmp /= num_samples;
  const int64_t head_idx = tmp % nheads;
  tmp /= nheads;
  const int64_t batch_idx = tmp % batch;
  const int64_t query_idx = tmp / batch;

  int64_t leaf_idx = (((query_idx * batch) + batch_idx) * nheads + head_idx) * num_samples + sample_idx;
  leaf_idx = static_cast<int64_t>(sampled_indices[leaf_idx]);
  if (leaf_idx < 0) {
    leaf_idx = 0;
  } else if (leaf_idx >= num_leaves) {
    leaf_idx = num_leaves - 1;
  }

  const int64_t grad_out_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * width + d;
  const int64_t weight_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * num_samples + sample_idx;
  const int64_t grad_v_offset = (((leaf_idx * batch) + batch_idx) * nheads + head_idx) * width + d;

  gpuAtomicAddNoReturn(
      grad_v + grad_v_offset,
      static_cast<out_t>(
          attn_weights[weight_offset] * static_cast<float>(grad_output[grad_out_offset])));
}

}  // namespace

template <typename scalar_t, typename out_t, typename index_t>
void launch_scatter_weighted_grad_v_forward_kernel(
  const torch::Tensor& grad_output,
  const torch::Tensor& sampled_indices,
  const torch::Tensor& attn_weights,
  torch::Tensor& grad_v,
  int64_t num_leaves,
  int64_t batch,
  int64_t nheads,
  int64_t width,
  int64_t num_queries,
  int64_t num_samples,
  int blocks,
  int threads) {
  scatter_weighted_grad_v_forward_kernel<scalar_t, out_t, index_t><<<
    blocks,
    threads,
    0,
    at::cuda::getDefaultCUDAStream()>>>(
    grad_output.data_ptr<scalar_t>(),
    sampled_indices.data_ptr<index_t>(),
    attn_weights.data_ptr<float>(),
    grad_v.data_ptr<out_t>(),
    num_leaves,
    batch,
    nheads,
    width,
    num_queries,
    num_samples);
}

torch::Tensor scatter_weighted_grad_v_forward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& sampled_indices,
    const torch::Tensor& attn_weights,
    int64_t num_leaves) {
  c10::cuda::CUDAGuard device_guard(grad_output.device());

  const auto num_queries = grad_output.size(0);
  const auto batch = grad_output.size(1);
  const auto nheads = grad_output.size(2);
  const auto width = grad_output.size(3);
  const auto num_samples = sampled_indices.size(3);

  auto grad_v = torch::zeros(
      {num_leaves, batch, nheads, width},
    grad_output.options());

  constexpr int threads = 256;
  const int64_t total = num_queries * batch * nheads * num_samples * width;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "scatter_weighted_grad_v_forward_cuda",
      [&] {
        if (sampled_indices.scalar_type() == torch::kInt32) {
          launch_scatter_weighted_grad_v_forward_kernel<scalar_t, scalar_t, int32_t>(
              grad_output,
              sampled_indices,
              attn_weights,
              grad_v,
              num_leaves,
              batch,
              nheads,
              width,
              num_queries,
              num_samples,
              blocks,
              threads);
        } else {
          launch_scatter_weighted_grad_v_forward_kernel<scalar_t, scalar_t, int64_t>(
              grad_output,
              sampled_indices,
              attn_weights,
              grad_v,
              num_leaves,
              batch,
              nheads,
              width,
              num_queries,
              num_samples,
              blocks,
              threads);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_v;
}