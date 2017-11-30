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

__global__ void ComputeLocalMean(const float3* lab_in,
                                 float3* mean_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT)
    return;
  float3 mean = make_float3(0.0, 0.0, 0.0);
  int n = 0;
  for (int i = 0; i < min(WINDOW_SIZE, IMAGE_HEIGHT - y); ++i) {
    int row_offset = ((y + i) * IMAGE_HEIGHT) + x;
    for (int j = 0; j < min(WINDOW_SIZE, IMAGE_WIDTH - x); ++j) {
      float3 lab = lab_in[row_offset + j];
      mean = make_float3(mean.x + lab.x,
                         mean.y + lab.y,
                         mean.z + lab.z);
      ++n;
    }
  }
  float n_float = __int2float_rn(n);
  mean = make_float3(mean.x / n_float,
                     mean.y / n_float,
                     mean.z / n_float);
  mean_out[(y * IMAGE_WIDTH) + x] = mean;
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
    int row_offset = ((y + i) * IMAGE_HEIGHT) + x;
    for (int j = 0; j < min(WINDOW_SIZE, IMAGE_WIDTH - x); ++j) {
      float3 lab = lab_in[row_offset + j];
      float3 mean = mean_in[row_offset + j];
      float3 del = make_float3(lab.x - mean.x,
                               lab.y - mean.y,
                               lab.z - mean.z);
      std_dev = make_float3(std_dev.x + (del.x * del.x),
                            std_dev.y + (del.y * del.y),
                            std_dev.z + (del.z * del.z));
      ++n;
    }
  }
  n = max(1, n - 1);
  float n_float = __int2float_rn(n);
  std_dev = make_float3(std_dev.x / n_float,
                        std_dev.y / n_float,
                        std_dev.z / n_float);
  stddevsq_out[(y * IMAGE_WIDTH) + x] = std_dev;
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
    int row_offset = ((y + i) * IMAGE_HEIGHT) + x;
    for (int j = 0; j < min(WINDOW_SIZE, IMAGE_WIDTH - x); ++j) {
      float3 lab_a = lab_a_in[row_offset + j];
      float3 mean_a = mean_a_in[row_offset + j];
      float3 del_a = make_float3(lab_a.x - mean_a.x,
                                 lab_a.y - mean_a.y,
                                 lab_a.z - mean_a.z);

      float3 lab_b = lab_b_in[row_offset + j];
      float3 mean_b = mean_b_in[row_offset + j];
      float3 del_b = make_float3(lab_b.x - mean_b.x,
                                 lab_b.y - mean_b.y,
                                 lab_b.z - mean_b.z);
      cov = make_float3(cov.x + (del_a.x * del_b.x),
                        cov.y + (del_a.y * del_b.y),
                        cov.z + (del_a.z * del_b.z));
    }
  }
  n = max(1, n - 1);
  float n_float = __int2float_rn(n);
  cov = make_float3(cov.x / n_float,
                    cov.y / n_float,
                    cov.z / n_float);
  cov_ab_out[(y * IMAGE_WIDTH) + x] = cov;
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
  float3 mean_a = mean_a_in[(y * IMAGE_WIDTH) + x];
  float3 stddevsq_a = stddevsq_a_in[(y * IMAGE_WIDTH) + x];
  float3 mean_b = mean_b_in[(y * IMAGE_WIDTH) + x];
  float3 stddevsq_b = stddevsq_b_in[(y * IMAGE_WIDTH) + x];
  float3 cov_ab = cov_ab_in[(y * IMAGE_WIDTH) + x];
  float3 ssim = make_float3(
      (((2.0 * mean_a.x * mean_b.x) + C1) * ((2.0 * cov_ab.x) + C2)) /
          (((mean_a.x * mean_a.x) + (mean_b.x * mean_b.x) + C1) *
           (stddevsq_a.x + stddevsq_b.x + C2)),
      (((2.0 * mean_a.y * mean_b.y) + C1) * ((2.0 * cov_ab.y) + C2)) /
          (((mean_a.y * mean_a.y) + (mean_b.y * mean_b.y) + C1) *
           (stddevsq_a.y + stddevsq_b.y + C2)),
      (((2.0 * mean_a.z * mean_b.z) + C1) * ((2.0 * cov_ab.z) + C2)) /
          (((mean_a.z * mean_a.z) + (mean_b.z * mean_b.z) + C1) *
           (stddevsq_a.z + stddevsq_b.z + C2)));

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

// 120 floats in, 1 float out. 60 threads, 60 floats in shared storage.
__global__ void ComputeFinalSum(const float* block_sum_in, float* sum_out) {
  __shared__ float sum_shared[60];
  int thread_id = threadIdx.x;
  sum_shared[thread_id] = block_sum_in[thread_id * 2] +
      block_sum_in[(thread_id * 2) + 1];
  __syncthreads();

  if (thread_id < 30) {
    sum_shared[thread_id] = sum_shared[thread_id + 30];
  }
  __syncthreads();

  if (thread_id == 0) {
    float sum_a = sum_shared[0];
    sum_a += sum_shared[ 1];
    sum_a += sum_shared[ 2];
    sum_a += sum_shared[ 3];
    sum_a += sum_shared[ 4];
    sum_a += sum_shared[ 5];
    sum_a += sum_shared[ 6];
    sum_a += sum_shared[ 7];
    sum_a += sum_shared[ 8];
    sum_a += sum_shared[ 9];
    sum_a += sum_shared[10];
    sum_a += sum_shared[11];
    sum_a += sum_shared[12];
    sum_a += sum_shared[13];
    sum_a += sum_shared[14];
    sum_shared[0] = sum_a;
  } else if (thread_id == 1) {
    float sum_b = sum_shared[15];
    sum_b += sum_shared[16];
    sum_b += sum_shared[17];
    sum_b += sum_shared[18];
    sum_b += sum_shared[19];
    sum_b += sum_shared[20];
    sum_b += sum_shared[21];
    sum_b += sum_shared[22];
    sum_b += sum_shared[23];
    sum_b += sum_shared[24];
    sum_b += sum_shared[25];
    sum_b += sum_shared[26];
    sum_b += sum_shared[27];
    sum_b += sum_shared[28];
    sum_b += sum_shared[29];
    sum_shared[15] = sum_b;
  }
  __syncthreads();

  if (thread_id == 0) {
    *sum_out = sum_shared[0] + sum_shared[15];
  }
}

}  // namespace vcsmc
