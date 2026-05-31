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

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return __shfl_sync(0xffffffffU, value, 0);
}

int64_t compute_log_n_host(int64_t num_leaves) {
  int64_t log_n = 0;
  while (num_leaves > 1) {
    ++log_n;
    num_leaves >>= 1;
  }
  return log_n;
}

template <typename scalar_t, typename acc_t, typename index_t>
__global__ void accumulate_qk_non_causal_all_depths_prepared_query_tiled_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ grad_output,
    const index_t* __restrict__ sampled_indices,
    const float* __restrict__ attn_weights,
    const uint8_t* __restrict__ packed_paths,
    float* __restrict__ grad_q_out,
    float* __restrict__ grad_k_out,
    int64_t num_leaves,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples,
    int64_t path_bytes,
    int64_t log_n,
    float max_logit) {
  extern __shared__ float shared[];
  float* q_shared = shared;
  float* grad_out_shared = q_shared + width;
  float* grad_q_partials = grad_out_shared + width;
  float* sample_attn_weight = grad_q_partials + num_samples * width;
  float* sample_grad_attn = sample_attn_weight + num_samples;
  float* sample_grad_log_probs = sample_grad_attn + num_samples;

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int64_t qbh_idx = static_cast<int64_t>(blockIdx.x);
  const int64_t total_qbh = num_queries * batch * nheads;
  if (qbh_idx >= total_qbh) {
    return;
  }

  int64_t tmp = qbh_idx;
  const int64_t head_idx = tmp % nheads;
  tmp /= nheads;
  const int64_t batch_idx = tmp % batch;
  const int64_t query_idx = tmp / batch;

  const int64_t q_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * width;
  for (int64_t d = threadIdx.x; d < width; d += blockDim.x) {
    q_shared[d] = static_cast<float>(q[q_offset + d]);
    grad_out_shared[d] = static_cast<float>(grad_output[q_offset + d]);
  }
  for (int64_t idx = threadIdx.x; idx < num_samples * width; idx += blockDim.x) {
    grad_q_partials[idx] = 0.0f;
  }
  if (threadIdx.x < num_samples) {
    sample_attn_weight[threadIdx.x] = 0.0f;
    sample_grad_attn[threadIdx.x] = 0.0f;
    sample_grad_log_probs[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  if (warp_id < num_samples) {
    const int64_t sample_offset = qbh_idx * num_samples + warp_id;
    int64_t leaf_idx = static_cast<int64_t>(sampled_indices[sample_offset]);
    if (leaf_idx < 0) {
      leaf_idx = 0;
    } else if (leaf_idx >= num_leaves) {
      leaf_idx = num_leaves - 1;
    }

    if (lane == 0) {
      sample_attn_weight[warp_id] = attn_weights[sample_offset];
    }

    const int64_t v_offset = (((leaf_idx * batch) + batch_idx) * nheads + head_idx) * width;
    float grad_attn_partial = 0.0f;
    for (int64_t d = lane; d < width; d += 32) {
      grad_attn_partial += grad_out_shared[d] * static_cast<float>(v[v_offset + d]);
    }
    const float grad_attn = warp_reduce_sum(grad_attn_partial);
    if (lane == 0) {
      sample_grad_attn[warp_id] = grad_attn;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    float grad_center = 0.0f;
    for (int64_t sample = 0; sample < num_samples; ++sample) {
      grad_center += sample_attn_weight[sample] * sample_grad_attn[sample];
    }
    for (int64_t sample = 0; sample < num_samples; ++sample) {
      sample_grad_log_probs[sample] =
          sample_attn_weight[sample] * (sample_grad_attn[sample] - grad_center);
    }
  }
  __syncthreads();

  if (warp_id < num_samples) {
    const int64_t sample_offset = qbh_idx * num_samples + warp_id;
    const int64_t packed_offset = sample_offset * path_bytes;
    const float sample_grad_log_prob = sample_grad_log_probs[warp_id];

    int64_t node_idx = 0;
    for (int64_t depth = 0; depth < log_n; ++depth) {
      int64_t bit = 0;
      if (lane == 0) {
        const uint8_t packed_byte = packed_paths[packed_offset + (depth >> 3)];
        bit = static_cast<int64_t>((packed_byte >> (depth & 7)) & 1U);
      }
      bit = __shfl_sync(0xffffffffU, bit, 0);
      const float branch_sign = bit == 0 ? 1.0f : -1.0f;
      const int64_t k_offset = (((node_idx * batch) + batch_idx) * nheads + head_idx) * width;

      float dot_partial = 0.0f;
      for (int64_t d = lane; d < width; d += 32) {
        dot_partial += q_shared[d] * static_cast<float>(k[k_offset + d]);
      }
      float dot = warp_reduce_sum(dot_partial);
      if (dot > max_logit) {
        dot = max_logit;
      } else if (dot < -max_logit) {
        dot = -max_logit;
      }

      const float grad_logit = branch_sign * sigmoid_forward(-branch_sign * dot) * sample_grad_log_prob;
      for (int64_t d = lane; d < width; d += 32) {
        grad_q_partials[warp_id * width + d] += grad_logit * static_cast<float>(k[k_offset + d]);
        atomicAdd(grad_k_out + k_offset + d, grad_logit * q_shared[d]);
      }

      node_idx = 2 * node_idx + 1 + bit;
    }
  }

  __syncthreads();
  for (int64_t d = threadIdx.x; d < width; d += blockDim.x) {
    float grad_q = 0.0f;
    for (int64_t sample = 0; sample < num_samples; ++sample) {
      grad_q += grad_q_partials[sample * width + d];
    }
    grad_q_out[q_offset + d] = grad_q;
  }
}

template <typename scalar_t, typename acc_t, typename index_t>
void launch_accumulate_qk_non_causal_all_depths_prepared_query_tiled_kernel(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& grad_output,
    const torch::Tensor& sampled_indices,
    const torch::Tensor& attn_weights,
    const torch::Tensor& packed_paths,
    const torch::Tensor& grad_q_out,
    const torch::Tensor& grad_k_out,
    int64_t num_leaves,
    int64_t batch,
    int64_t nheads,
    int64_t width,
    int64_t num_queries,
    int64_t num_samples,
    int64_t path_bytes,
    int64_t log_n,
    float max_logit) {
  const int blocks = static_cast<int>(num_queries * batch * nheads);
  const int threads = static_cast<int>(num_samples * 32);
  const size_t shared_bytes =
      static_cast<size_t>(2 * width + num_samples * width + 3 * num_samples) * sizeof(float);

  accumulate_qk_non_causal_all_depths_prepared_query_tiled_kernel<scalar_t, acc_t, index_t><<<
      blocks,
      threads,
      shared_bytes,
      at::cuda::getDefaultCUDAStream()>>>(
      q.data_ptr<scalar_t>(),
      k.data_ptr<scalar_t>(),
      v.data_ptr<scalar_t>(),
      grad_output.data_ptr<scalar_t>(),
      sampled_indices.data_ptr<index_t>(),
      attn_weights.data_ptr<float>(),
      packed_paths.data_ptr<uint8_t>(),
      grad_q_out.data_ptr<float>(),
      grad_k_out.data_ptr<float>(),
      num_leaves,
      batch,
      nheads,
      width,
      num_queries,
      num_samples,
      path_bytes,
      log_n,
      max_logit);
}

}  // namespace

void accumulate_qk_non_causal_all_depths_prepared_inplace_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& grad_output,
    const torch::Tensor& sampled_indices,
    const torch::Tensor& attn_weights,
    const torch::Tensor& packed_paths,
    double max_logit,
    const torch::Tensor& grad_q_out,
    const torch::Tensor& grad_k_out) {
  c10::cuda::CUDAGuard device_guard(q.device());

  const auto num_queries = q.size(0);
  const auto batch = q.size(1);
  const auto nheads = q.size(2);
  const auto width = q.size(3);
  const auto num_leaves = v.size(0);
  const auto num_samples = sampled_indices.size(3);
  const auto path_bytes = packed_paths.size(4);
  const auto log_n = compute_log_n_host(num_leaves);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "accumulate_qk_non_causal_all_depths_prepared_inplace_cuda",
      [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        if (sampled_indices.scalar_type() == torch::kInt32) {
          launch_accumulate_qk_non_causal_all_depths_prepared_query_tiled_kernel<
              scalar_t,
              acc_t,
              int32_t>(
              q,
              k,
              v,
              grad_output,
              sampled_indices,
              attn_weights,
              packed_paths,
              grad_q_out,
              grad_k_out,
              num_leaves,
              batch,
              nheads,
              width,
              num_queries,
              num_samples,
              path_bytes,
              log_n,
              static_cast<float>(max_logit));
        } else {
          launch_accumulate_qk_non_causal_all_depths_prepared_query_tiled_kernel<
              scalar_t,
              acc_t,
              int64_t>(
              q,
              k,
              v,
              grad_output,
              sampled_indices,
              attn_weights,
              packed_paths,
              grad_q_out,
              grad_k_out,
              num_leaves,
              batch,
              nheads,
              width,
              num_queries,
              num_samples,
              path_bytes,
              log_n,
              static_cast<float>(max_logit));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}