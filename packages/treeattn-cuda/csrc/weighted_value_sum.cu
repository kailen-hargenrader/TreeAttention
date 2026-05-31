#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t, typename acc_t, typename index_t>
__global__ void weighted_value_sum_forward_kernel(
    const scalar_t* __restrict__ v,
  const index_t* __restrict__ sampled_indices,
    const float* __restrict__ attn_weights,
    scalar_t* __restrict__ out,
    int64_t num_leaves,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples) {
  int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_queries * batch * nheads * width;
  if (linear_idx >= total) {
    return;
  }

  int64_t d = linear_idx % width;
  int64_t tmp = linear_idx / width;
  int64_t h = tmp % nheads;
  tmp /= nheads;
  int64_t b = tmp % batch;
  int64_t l = tmp / batch;

  acc_t acc = static_cast<acc_t>(0);
  int64_t sample_base = (((l * batch) + b) * nheads + h) * num_samples;
  for (int64_t s = 0; s < num_samples; ++s) {
    int64_t leaf_idx = static_cast<int64_t>(sampled_indices[sample_base + s]);
    if (leaf_idx < 0) {
      leaf_idx = 0;
    } else if (leaf_idx >= num_leaves) {
      leaf_idx = num_leaves - 1;
    }
    int64_t value_offset = (((leaf_idx * batch) + b) * nheads + h) * width + d;
    acc += static_cast<acc_t>(attn_weights[sample_base + s]) * static_cast<acc_t>(v[value_offset]);
  }

  out[linear_idx] = static_cast<scalar_t>(acc);
}

}  // namespace

template <typename scalar_t, typename acc_t>
void launch_weighted_value_sum_forward_kernel(
    const torch::Tensor& v,
    const torch::Tensor& sampled_indices,
    const torch::Tensor& attn_weights,
    torch::Tensor& out,
    int64_t num_leaves,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples,
    int blocks,
    int threads) {
  if (sampled_indices.scalar_type() == torch::kInt32) {
    weighted_value_sum_forward_kernel<scalar_t, acc_t, int32_t><<<
        blocks,
        threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
        v.data_ptr<scalar_t>(),
        sampled_indices.data_ptr<int32_t>(),
        attn_weights.data_ptr<float>(),
        out.data_ptr<scalar_t>(),
        num_leaves,
        batch,
        nheads,
        width,
        num_queries,
        num_samples);
    return;
  }

  weighted_value_sum_forward_kernel<scalar_t, acc_t, int64_t><<<
      blocks,
      threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
      v.data_ptr<scalar_t>(),
      sampled_indices.data_ptr<int64_t>(),
      attn_weights.data_ptr<float>(),
      out.data_ptr<scalar_t>(),
      num_leaves,
      batch,
      nheads,
      width,
      num_queries,
      num_samples);
}

torch::Tensor weighted_value_sum_forward_cuda(
    const torch::Tensor& v,
    const torch::Tensor& sampled_indices,
    const torch::Tensor& attn_weights) {
  c10::cuda::CUDAGuard device_guard(v.device());

  auto num_leaves = v.size(0);
  auto batch = v.size(1);
  auto nheads = v.size(2);
  auto width = v.size(3);
  auto num_queries = sampled_indices.size(0);
  auto num_samples = sampled_indices.size(3);

  auto out = torch::zeros({num_queries, batch, nheads, width}, v.options());
  auto total = num_queries * batch * nheads * width;
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      v.scalar_type(),
      "weighted_value_sum_forward_cuda",
      [&] {
        using acc_t = at::acc_type<scalar_t, true>;
      launch_weighted_value_sum_forward_kernel<scalar_t, acc_t>(
        v,
        sampled_indices,
        attn_weights,
        out,
        num_leaves,
        batch,
        nheads,
        width,
        num_queries,
        num_samples,
        blocks,
        threads);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}