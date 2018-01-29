#include "mssim.h"

#include "color.h"

namespace vcsmc {

__global__ void LabaToNormalizedL(const float4* laba_in,
                                  float* nl_out) {
  size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  size_t index = (y * kTargetFrameWidthPixels) + x;

  if (x >= kTargetFrameWidthPixels || y >= kFrameHeightPixels) {
    return;
  }

  float4 laba = laba_in[index];
  nl_out[index] = laba.x / kMaxLab;
}

__global__ void ComputeMean(const float* nl_in,
                            float* mean_out) {
  size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  size_t index = (y * kTargetFrameWidthPixels) + x;

  if (x >= kTargetFrameWidthPixels || y >= kFrameHeightPixels) {
    return;
  }

  float sum = 0.0;
  float n = 0.0;
  for (size_t i = 0; i < min(kWindowSize, kFrameHeightPixels - y); ++i) {
    size_t row_offset = index + (i * kTargetFrameWidthPixels);
    for (size_t j = 0; j < min(kWindowSize, kTargetFrameWidthPixels - x); ++j) {
      sum = sum + nl_in[row_offset + j];
      n = n + 1.0;
    }
  }

  n = max(n, 1.0);
  mean_out[index] = sum / n;
}

__global__ void ComputeVariance(const float* nl_in,
                                const float* mean_in,
                                float* variance_out) {
  size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  size_t index = (y * kTargetFrameWidthPixels) + x;

  if (x >= kTargetFrameWidthPixels || y >= kFrameHeightPixels) {
    return;
  }

  float mean = mean_in[index];
  float sum = 0.0;
  float n = 0.0;
  for (size_t i = 0; i < min(kWindowSize, kFrameHeightPixels - y); ++i) {
    size_t row_offset = index + (i * kTargetFrameWidthPixels);
    for (size_t j = 0; j < min(kWindowSize, kTargetFrameWidthPixels - x); ++j) {
      float del = nl_in[row_offset + j] - mean;
      sum = sum + (del * del);
      n = n + 1.0;
    }
  }

  float n_minus_one = max(n - 1.0, 1.0);
  variance_out[index] = sum / n_minus_one;
}

__global__ void ComputeCovariance(const float* nl_a_in,
                                  const float* mean_a_in,
                                  const float* nl_b_in,
                                  const float* mean_b_in,
                                  float* cov_ab_out) {
  size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  size_t index = (y * kTargetFrameWidthPixels) + x;

  if (x >= kTargetFrameWidthPixels || y >= kFrameHeightPixels) {
    return;
  }

  float mean_a = mean_a_in[index];
  float mean_b = mean_b_in[index];
  float sum = 0.0;
  float n = 0.0;
  for (size_t i = 0; i < min(kWindowSize, kFrameHeightPixels - y); ++i) {
    size_t row_offset = index + (i * kTargetFrameWidthPixels);
    for (size_t j = 0; j < min(kWindowSize, kTargetFrameWidthPixels - x); ++j) {
      sum = sum + ((nl_a_in[row_offset + j] - mean_a) *
                   (nl_b_in[row_offset + j] - mean_b));
      n = n + 1.0;
    }
  }

  float n_minus_one = max(n - 1.0, 1.0);
  cov_ab_out[index] = sum / n_minus_one;
}


const float kC1 = 0.0001;
const float kC2 = 0.0009;

__global__ void ComputeSSIM(const float* mean_a_in,
                            const float* variance_a_in,
                            const float* mean_b_in,
                            const float* variance_b_in,
                            const float* cov_ab_in,
                            float* ssim_out) {
  size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  size_t index = (y * kTargetFrameWidthPixels) + x;

  if (x >= kTargetFrameWidthPixels || y >= kFrameHeightPixels) {
    return;
  }

  float mean_a = mean_a_in[index];
  float variance_a = variance_a_in[index];
  float mean_b = mean_b_in[index];
  float variance_b = variance_b_in[index];
  float cov_ab = cov_ab_in[index];
  float ssim = (((2.0 * mean_a * mean_b) + kC1) * ((2.0 * cov_ab) + kC2)) /
                (((mean_a * mean_a) + (mean_b * mean_b) + kC1) *
                 (variance_a + variance_b + kC2));

  ssim_out[index] = fabs(ssim);
}

// Parallel summing algorithm. Blocks at 32x32 for 1024 threads per block, with
// 10x6 blocks, returning 60 floats for final summation on host.
//
// * * * * * * * * * * * * * * * *  if (thread_id < 8) {
// | | | | | | | | | | | | | | | |    block_shared[thread_id] +=
// +-|-|-|-|-|-|-|-/ | | | | | | |       block_shared[thread_id + 8];
// | +-|-|-|-|-|-|---/ | | | | | |  }
// | | +-|-|-|-|-|-----/ | | | | |
// | | | +-|-|-|-|-------/ | | | |
// | | | | +-|-|-|-------- / | | |
// | | | | | +-|-|-----------/ | |
// | | | | | | +-|-------------/ |
// | | | | | | | +---------------/
// | | | | | | | |
// * * * * * * * *                  if (thread_id < 4) {
// | | | | | | | |                    block_shared[thread_id] +=
// +-|-|-|-/ | | |                        block_shared[thread_id + 4];
// | +-|-|---/ | |                  }
// | | +-|-----/ |
// | | | +-------/
// | | | |
// * * * *                          if (thread_id < 2) {
// | | | |                            block_shared[thread_id] +=
// +-|-/ |                                block_shared[thread_id + 2];
// | +---/                          }
// | |
// * *                              if (thread_id == 0) {
// | |                                block_shared[0] += block_shared[1];
// +-/                              }
// |
// *
__global__ void ComputeBlockSum(const float* ssim_in, float* block_sum_out) {
  __shared__ float block_shared[1024];
  size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  size_t thread_id = (threadIdx.y * blockDim.x) + threadIdx.x;
  size_t pixel_offset = (y * kTargetFrameWidthPixels) + x;
  block_shared[thread_id] = ssim_in[pixel_offset];
  __syncthreads();

  if (thread_id < 512) {
    block_shared[thread_id] += block_shared[thread_id + 512];
  }
  __syncthreads();

  if (thread_id < 256) {
    block_shared[thread_id] += block_shared[thread_id + 256];
  }
  __syncthreads();

  if (thread_id < 128) {
    block_shared[thread_id] += block_shared[thread_id + 128];
  }
  __syncthreads();

  if (thread_id < 64) {
    block_shared[thread_id] += block_shared[thread_id + 64];
  }
  __syncthreads();

  if (thread_id < 32) {
    block_shared[thread_id] += block_shared[thread_id + 32];
  }
  __syncthreads();

  if (thread_id < 16) {
    block_shared[thread_id] += block_shared[thread_id + 16];
  }
  __syncthreads();

  if (thread_id < 8) {
    block_shared[thread_id] += block_shared[thread_id + 8];
  }
  __syncthreads();

  if (thread_id < 4) {
    block_shared[thread_id] += block_shared[thread_id + 4];
  }
  __syncthreads();

  if (thread_id < 2) {
    block_shared[thread_id] += block_shared[thread_id + 2];
  }
  __syncthreads();

  if (thread_id == 0) {
    size_t block_index = (blockIdx.y * gridDim.x) + blockIdx.x;
    block_sum_out[block_index] = block_shared[0] + block_shared[1];
  }
}

}  // namespace vcsmc
