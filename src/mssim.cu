#include "mssim.h"

namespace vcsmc {

#define WINDOW_SIZE 8
#define C1 ((0.01 * 100.0) * (0.01 * 100.0))
#define C2 ((0.03 * 100.0) * (0.03 * 100.0))
#define WEIGHT_L 0.5
#define WEIGHT_A 0.25
#define WEIGHT_B 0.25
#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 192
#define PADDED_IMAGE_WIDTH (IMAGE_WIDTH + WINDOW_SIZE)
#define PADDED_IMAGE_HEIGHT (IMAGE_HEIGHT + WINDOW_SIZE)

__global__ void ComputeLocalMean(const float4* nyuv_in,
                                 float4* mean_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  float4 mean = make_float4(0.0, 0.0, 0.0, 0.0);

  // If we are running on the padding then mark the output as invalid too.
  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) {
    if (x < PADDED_IMAGE_WIDTH && y < PADDED_IMAGE_HEIGHT) {
      mean_out[(y * PADDED_IMAGE_WIDTH) + x] = mean;
    }
    return;
  }

  for (int i = 0; i < WINDOW_SIZE; ++i) {
    int row_offset = ((y + i) * PADDED_IMAGE_WIDTH) + x;
    for (int j = 0; j < WINDOW_SIZE; ++j) {
      float4 nyuv = nyuv_in[row_offset + j];
      // fourth element to be 1.0 in valid elements, 0.0 in padding.
      mean = make_float4(mean.x + nyuv.x,
                         mean.y + nyuv.y,
                         mean.z + nyuv.z,
                         mean.w + nyuv.w);
    }
  }

  mean = make_float4(mean.x / mean.w,
                     mean.y / mean.w,
                     mean.z / mean.w,
                     1.0);
  mean_out[(y * PADDED_IMAGE_WIDTH) + x] = mean;
}

__global__ void ComputeLocalStdDevSquared(const float4* nyuv_in,
                                          const float4* mean_in,
                                          float4* stddevsq_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  float4 std_dev = make_float4(0.0, 0.0, 0.0, 0.0);

  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) {
    if (x < PADDED_IMAGE_WIDTH && y < PADDED_IMAGE_HEIGHT) {
      stddevsq_out[(y * PADDED_IMAGE_WIDTH) + x] = std_dev;
    }
    return;
  }

  float4 mean = mean_in[(y * PADDED_IMAGE_WIDTH) + x];

  for (int i = 0; i < WINDOW_SIZE; ++i) {
    int row_offset = ((y + i) * PADDED_IMAGE_WIDTH) + x;
    for (int j = 0; j < WINDOW_SIZE; ++j) {
      float4 nyuv = nyuv_in[row_offset + j];
      float4 del = make_float4(nyuv.x - mean.x,
                               nyuv.y - mean.y,
                               nyuv.z - mean.z,
                               0.0);
      std_dev = make_float4(std_dev.x + (del.x * del.x),
                            std_dev.y + (del.y * del.y),
                            std_dev.z + (del.z * del.z),
                            std_dev.w + nyuv.w);
    }
  }

  float n_minus_one = min(std_dev.w - 1.0, 1.0);
  std_dev = make_float4(std_dev.x / n_minus_one,
                        std_dev.y / n_minus_one,
                        std_dev.z / n_minus_one,
                        1.0);
  stddevsq_out[(y * PADDED_IMAGE_WIDTH) + x] = std_dev;
}

__global__ void ComputeLocalCovariance(const float4* nyuv_a_in,
                                       const float4* mean_a_in,
                                       const float4* nyuv_b_in,
                                       const float4* mean_b_in,
                                       float4* cov_ab_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  float4 cov = make_float4(0.0, 0.0, 0.0, 0.0);
  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) {
    if (x < PADDED_IMAGE_WIDTH && y < PADDED_IMAGE_HEIGHT) {
      cov_ab_out[(y * PADDED_IMAGE_WIDTH) + x] = cov;
    }
    return;
  }

  float4 mean_a = mean_a_in[(y * PADDED_IMAGE_WIDTH) + x];
  float4 mean_b = mean_b_in[(y * PADDED_IMAGE_WIDTH) + x];

  for (int i = 0; i < WINDOW_SIZE; ++i) {
    int row_offset = ((y + i) * PADDED_IMAGE_WIDTH) + x;
    for (int j = 0; j < WINDOW_SIZE; ++j) {
      float4 nyuv_a = nyuv_a_in[row_offset + j];
      float4 del_a = make_float4(nyuv_a.x - mean_a.x,
                                 nyuv_a.y - mean_a.y,
                                 nyuv_a.z - mean_a.z,
                                 0.0);

      float4 nyuv_b = nyuv_b_in[row_offset + j];
      float4 del_b = make_float4(nyuv_b.x - mean_b.x,
                                 nyuv_b.y - mean_b.y,
                                 nyuv_b.z - mean_b.z,
                                 0.0);

      cov = make_float4(cov.x + (del_a.x * del_b.x),
                        cov.y + (del_a.y * del_b.y),
                        cov.z + (del_a.z * del_b.z),
                        cov.w + nyuv_a.w);
    }
  }

  float n_minus_one = min(cov.w - 1.0, 1.0);
  cov = make_float4(cov.x / n_minus_one,
                    cov.y / n_minus_one,
                    cov.z / n_minus_one,
                    1.0);
  cov_ab_out[(y * PADDED_IMAGE_WIDTH) + x] = cov;
}

__global__ void ComputeSSIM(const float4* mean_a_in,
                            const float4* stddevsq_a_in,
                            const float4* mean_b_in,
                            const float4* stddevsq_b_in,
                            const float4* cov_ab_in,
                            float* ssim_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT)
    return;
  int pixel_offset = (y * PADDED_IMAGE_WIDTH) + x;
  float4 mean_a = mean_a_in[pixel_offset];
  float4 stddevsq_a = stddevsq_a_in[pixel_offset];
  float4 mean_b = mean_b_in[pixel_offset];
  float4 stddevsq_b = stddevsq_b_in[pixel_offset];
  float4 cov_ab = cov_ab_in[pixel_offset];
  float4 ssim = make_float4(
      (((2.0 * mean_a.x * mean_b.x) + C1) * ((2.0 * cov_ab.x) + C2)) /
          (((mean_a.x * mean_a.x) + (mean_b.x * mean_b.x) + C1) *
           (stddevsq_a.x + stddevsq_b.x + C2)),
      (((2.0 * mean_a.y * mean_b.y) + C1) * ((2.0 * cov_ab.y) + C2)) /
          (((mean_a.y * mean_a.y) + (mean_b.y * mean_b.y) + C1) *
           (stddevsq_a.y + stddevsq_b.y + C2)),
      (((2.0 * mean_a.z * mean_b.z) + C1) * ((2.0 * cov_ab.z) + C2)) /
          (((mean_a.z * mean_a.z) + (mean_b.z * mean_b.z) + C1) *
           (stddevsq_a.z + stddevsq_b.z + C2)),
      0.0);

  // Note we must calculate an unpadded offset for the ssim_out result, as
  // the output buffer needs no padding as there are no loops to unroll here.
  ssim_out[(y * IMAGE_WIDTH) + x] =
      (WEIGHT_L * fabsf(ssim.x)) +
      (WEIGHT_A * fabsf(ssim.y)) +
      (WEIGHT_B * fabsf(ssim.z));
}

// 320 * 192 = 61440 = 256 * 240
// Block size assumed to be 256, grid size 120. Each block needs 256 floats
// of shared storage. Output will be 120 floats.
#define SUM_BLOCK_SIZE 256

// With thanks to the excellent presentation on parallel sums by Mark Harris,
// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/
//    projects/reduction/doc/reduction.pdf
__global__ void ComputeBlockSum(const float* ssim_in, float* block_sum_out) {
  __shared__ float block_shared[SUM_BLOCK_SIZE];
  int thread_id = threadIdx.x;
  int global_id = (blockIdx.x * SUM_BLOCK_SIZE * 2) + thread_id;
  block_shared[thread_id] =
      ssim_in[global_id] + ssim_in[global_id + SUM_BLOCK_SIZE];
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
