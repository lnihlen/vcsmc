#include "mssim.h"

#include <algorithm>
#include <memory>

#include "constants.h"

namespace vcsmc {

#define WINDOW_SIZE 8
#define C1 ((0.01 * 100.0) * (0.01 * 100.0))
#define C2 ((0.03 * 100.0) * (0.03 * 100.0))
#define WEIGHT_L 0.5
#define WEIGHT_A 0.25
#define WEIGHT_B 0.25
#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 192

__global__ void ComputeLocalMean(const float3* lab_in,
                                 float3* mean_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT)
    return;
  float3 mean = make_float3(0.0, 0.0, 0.0);
  int n = 0;
  for (int i = 0; i < min(WINDOW_SIZE, IMAGE_HEIGHT - y); ++i) {
    for (int j = 0; j < min(WINDOW_SIZE, IMAGE_WIDTH - x); ++j) {
      mean += lab_in[j][i];
      ++n;
    }
  }
  mean = mean / __int2float_rn(n);
  mean_out[x][y] = mean;
}

__global__ void ComputeLocalStdDevSquared(const float3* lab_in,
                                          const float3* mean_in,
                                          float3* stddevsq_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT)
    return;
  float3 std_dev = make_float3(0.0, 0.0, 0.0);
  int n = 0;
  for (int i = 0; i < min(WINDOW_SIZE, IMAGE_HEIGHT - y); ++i) {
    for (int j = 0; j < min(WINDOW_SIZE, IMAGE_WIDTH - x); ++j) {
      float3 del = lab_in[j][i] - mean_in[j][i];
      std_dev += del * del;
      ++n;
    }
  }
  n = max(1, n - 1);
  std_dev = std_dev / __int2float_rn(n);
  stddevsq_out = std_dev;
}

__global__ void ComputeLocalCovariance(const float3* lab_a_in,
                                       const float3* mean_a_in,
                                       const float3* lab_b_in,
                                       const float3* mean_b_in,
                                       float3* cov_ab_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT)
    return;
  float3 cov = make_float3(0.0, 0.0, 0.0);
  int n = 0;
  for (int i = 0; i < min(WINDOW_SIZE, IMAGE_HEIGHT - y); ++i) {
    for (int j = 0; j < min(WINDOW_SIZE, IMAGE_WIDTH - x); ++j) {
      float3 del_a = lab_a_in[j][i] - mean_a_in[j][i];
      float3 del_b = lab_b_in[j][i] - mean_b_in[j][i];
      cov += del_a * del_b;
    }
  }
  n = max(1, n - 1);
  cov = cov / __int2float_rn(n);
  cov_ab_out[x][y] = cov;
}

__global__ void ComputeSSIM(const float3* mean_a_in,
                            const float3* stddevsq_a_in,
                            const float3* mean_b_in,
                            const float3* stddevsq_b_in,
                            const float3* cov_ab_in,
                            float* ssim_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT)
    return;
  float3 mean_a = mean_a_in[x][y];
  float3 mean_b = mean_b_in[x][y];
  float3 ssim =
      (((2.0 * mean_a * mean_b) + C1) * ((2 * cov_ab_in[x][y]) + C2)) /
          (((mean_a * mean_a) + (mean_b * mean_b) + C1) *
           (stddevsq_a[x][y] + stddevsq_b[x][y] + C2));
  ssim_out[(y * IMAGE_WIDTH) + x] =
      (WEIGHT_L * ssim.x) + (WEIGHT_A * ssim.y) + (WEIGHT_B * ssim.z);
}

// 320 * 192 = 61440 = 256 * 240
// Block size assumed to be 256, grid size 120. Each block needs 256 floats
// of shared storage. Output will be 120 floats.
#define SUM_BLOCK_SIZE 256

// With thanks to the excellent presentation on parallel sums by Mark Harris,
// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/
//    projects/reduction/doc/reduction.pdf
__global__ void ComputeBlockSum(const float* ssim_in, float* block_sum_out) {
  extern __shared__ float block_shared[SUM_BLOCK_SIZE];
  int thread_id = threadIdx.x;
  int global_id = (blockIdx.x * SUM_BLOCK_SIZE * 2) + thread_id;
  block_shared[thread_id] =
      ssim_in[global_id] + ssim_in[gloabl_id + SUM_BLOCK_SIZE];
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
    block_shared[thread_id] += block_shared[thread_id + 16];
    block_shared[thread_id] += block_shared[thread_id + 8];
    block_shared[thread_id] += block_shared[thread_id + 4];
    block_shared[thread_id] += block_shared[thread_id + 2];
    block_shared[thread_id] += block_shared[thread_id + 1];
  }

  if (thread_id == 0) {
    block_sum_out[blockIdx.x] = block_shared[0];
  }
}

}  // namespace vcsmc
