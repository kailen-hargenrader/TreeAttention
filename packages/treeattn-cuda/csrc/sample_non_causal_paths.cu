#include <ATen/AccumulateType.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TransformationHelper.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <curand_kernel.h>
#include <mutex>
#include <optional>
#include <tuple>

namespace {

constexpr uint32_t kDistributionBlockSize = 256;
constexpr uint32_t kDistributionUnrollFactor = 4;
constexpr uint32_t kDistributionMaxGeneratorOffsetsPerCall = 4;

__device__ __forceinline__ float log_sigmoid_forward(float x) {
  if (x >= 0.0f) {
    return -log1pf(expf(-x));
  }
  return x - log1pf(expf(x));
}

__device__ __forceinline__ float gumbel_from_full_exponential_draw(
    uint64_t seed,
    uint64_t offset,
    int64_t full_linear_idx,
    int64_t full_draw_stride) {
  const uint64_t thread_idx = static_cast<uint64_t>(full_linear_idx % full_draw_stride);
  const uint64_t bundle_idx = static_cast<uint64_t>(full_linear_idx / full_draw_stride);
  const uint64_t curand_call_idx = bundle_idx / kDistributionUnrollFactor;
  const uint32_t component_idx = static_cast<uint32_t>(bundle_idx % kDistributionUnrollFactor);

  curandStatePhilox4_32_10_t state;
  curand_init(seed, thread_idx, offset, &state);

  float4 uniform4;
  for (uint64_t call_idx = 0; call_idx <= curand_call_idx; ++call_idx) {
    uniform4 = curand_uniform4(&state);
  }

  const float uniform = (&uniform4.x)[component_idx];
  const float exponential_sample = at::transformation::exponential<float>(uniform, 1.0f);
  return -static_cast<float>(at::log(exponential_sample));
}

template <typename scalar_t, typename acc_t>
__global__ void sample_non_causal_step_kernel(
  const scalar_t* __restrict__ q,
  const scalar_t* __restrict__ k,
    int32_t* __restrict__ current_nodes,
    float* __restrict__ accumulated_log_probs,
    uint8_t* __restrict__ packed_paths,
    at::PhiloxCudaState gumbel_philox_args,
    int64_t full_draw_stride,
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

  const auto seeds = at::cuda::philox::unpack(gumbel_philox_args);
  const uint64_t philox_seed = std::get<0>(seeds);
  const uint64_t philox_offset = std::get<1>(seeds);

  int64_t tmp = linear_idx;
  const int64_t sample_idx = tmp % num_samples;
  tmp /= num_samples;
  const int64_t head_idx = tmp % nheads;
  tmp /= nheads;
  const int64_t batch_idx = tmp % batch;
  const int64_t query_idx = tmp / batch;

  const int64_t q_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * width;
  const int64_t current_offset = (((query_idx * batch) + batch_idx) * nheads + head_idx) * num_samples + sample_idx;
  const int64_t packed_offset = current_offset * path_bytes;

  const int64_t node_idx = static_cast<int64_t>(current_nodes[current_offset]);
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

  const float left_log_prob = log_sigmoid_forward(static_cast<float>(dot));
  const float right_log_prob = log_sigmoid_forward(static_cast<float>(-dot));
  const int64_t gumbel_offset = current_offset * 2;
  const float left_score = left_log_prob + gumbel_from_full_exponential_draw(
    philox_seed,
    philox_offset,
    gumbel_offset,
    full_draw_stride);
  const float right_score = right_log_prob + gumbel_from_full_exponential_draw(
    philox_seed,
    philox_offset,
    gumbel_offset + 1,
    full_draw_stride);
  const int32_t direction = right_score > left_score ? 1 : 0;
  const float chosen_log_prob = direction == 0 ? left_log_prob : right_log_prob;

  accumulated_log_probs[current_offset] += chosen_log_prob;
  packed_paths[packed_offset + (depth >> 3)] |= static_cast<uint8_t>(direction << (depth & 7));
  current_nodes[current_offset] = static_cast<int32_t>(2 * node_idx + 1 + direction);
}

int64_t compute_log_n(int64_t num_leaves) {
  int64_t log_n = 0;
  while (num_leaves > 1) {
    ++log_n;
    num_leaves >>= 1;
  }
  return log_n;
}

std::tuple<uint64_t, int64_t> compute_full_draw_rng_config(int64_t total_elements) {
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t blocks_per_sm = props->maxThreadsPerMultiProcessor / kDistributionBlockSize;
  const uint64_t max_grid_x = static_cast<uint64_t>(props->multiProcessorCount) * blocks_per_sm;
  const uint64_t requested_grid_x =
      (numel + kDistributionBlockSize - 1) / kDistributionBlockSize;
  const uint64_t grid_x = std::min(max_grid_x, requested_grid_x);
  const uint64_t counter_offset =
      ((numel - 1) /
           (static_cast<uint64_t>(kDistributionBlockSize) * grid_x *
            kDistributionUnrollFactor) +
       1) *
      kDistributionMaxGeneratorOffsetsPerCall;
  const int64_t full_draw_stride =
      static_cast<int64_t>(kDistributionBlockSize) * static_cast<int64_t>(grid_x);
  return std::make_tuple(counter_offset, full_draw_stride);
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
  (void)block_size;

  const auto num_queries = q.size(0);
  const auto batch = q.size(1);
  const auto nheads = q.size(2);
  const auto width = q.size(3);
  const auto k_len = k.size(0);
  const auto num_leaves = k_len + 1;
  const auto log_n = compute_log_n(num_leaves);
  const auto path_bytes = (log_n + 7) / 8;
  const auto total_samples = num_queries * batch * nheads * num_samples;
  const auto total_gumbel_values = total_samples * 2;

  auto current_nodes = torch::zeros(
      {num_queries, batch, nheads, num_samples},
      q.options().dtype(torch::kInt32));
  auto accumulated_log_probs = torch::zeros(
      {num_queries, batch, nheads, num_samples},
      q.options().dtype(torch::kFloat32));
  auto packed_paths = torch::zeros(
      {num_queries, batch, nheads, num_samples, path_bytes},
      q.options().dtype(torch::kUInt8));

  constexpr int threads = 256;
  const int blocks = static_cast<int>((total_samples + threads - 1) / threads);
  const auto [counter_offset, full_draw_stride] =
      compute_full_draw_rng_config(total_gumbel_values);
  std::optional<at::Generator> generator = std::nullopt;
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      generator,
      at::cuda::detail::getDefaultCUDAGenerator());

  for (int64_t depth = 0; depth < log_n; ++depth) {
    at::PhiloxCudaState rng_engine_inputs;
    {
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(counter_offset);
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "sample_non_causal_paths_forward_cuda",
        [&] {
          using acc_t = at::acc_type<scalar_t, true>;
          sample_non_causal_step_kernel<scalar_t, acc_t><<<
              blocks,
              threads,
              0,
              at::cuda::getDefaultCUDAStream()>>>(
              q.data_ptr<scalar_t>(),
              k.data_ptr<scalar_t>(),
              current_nodes.data_ptr<int32_t>(),
              accumulated_log_probs.data_ptr<float>(),
              packed_paths.data_ptr<uint8_t>(),
              rng_engine_inputs,
              full_draw_stride,
              k_len,
              batch,
              nheads,
              width,
              num_queries,
              num_samples,
              path_bytes,
              depth,
              static_cast<float>(max_logit));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  current_nodes.sub_(k_len);
  return std::make_tuple(current_nodes, accumulated_log_probs, packed_paths);
}