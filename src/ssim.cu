// ssim - given two images generates an error map and reports mean ssim for
// those two images.

#include <gflags/gflags.h>

#include "cuda.h"
#include "cuda_runtime.h"

#include "color.h"
#include "constants.h"
#include "cuda_utils.h"
#include "gray_map.h"
#include "image.h"
#include "image_file.h"
#include "mssim.h"

DEFINE_string(image_a, "", "Required - path to first image for comparison.");
DEFINE_string(image_b, "", "Required - path to second image for comparison.");
DEFINE_string(map_image, "", "Optional - path for grayscale output image for "
    "ssim map.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  if (!vcsmc::InitializeCuda(true)) return -1;

  std::unique_ptr<vcsmc::Image> image_a = vcsmc::LoadImage(FLAGS_image_a);
  if (!image_a) {
    fprintf(stderr, "error opening image_a %s\n", FLAGS_image_a.c_str());
    return -1;
  }
  std::unique_ptr<vcsmc::Image> image_b = vcsmc::LoadImage(FLAGS_image_b);
  if (!image_b) {
    fprintf(stderr, "error opening image_b %s\n", FLAGS_image_b.c_str());
    return -1;
  }

  if (image_a->width() != vcsmc::kTargetFrameWidthPixels ||
      image_a->height() != vcsmc::kFrameHeightPixels ||
      image_b->width() != vcsmc::kTargetFrameWidthPixels ||
      image_b->height() != vcsmc::kFrameHeightPixels) {
    fprintf(stderr, "image size mismatch.\n");
    return -1;
  }

  std::unique_ptr<float> image_a_laba = vcsmc::ImageToLabaArray(image_a.get());
  std::unique_ptr<float> image_b_laba = vcsmc::ImageToLabaArray(image_b.get());

  float4* image_a_device;
  cudaMalloc(&image_a_device, vcsmc::kLabaBufferSizeBytes);
  float4* image_b_device;
  cudaMalloc(&image_b_device, vcsmc::kLabaBufferSizeBytes);

  cudaMemcpy(image_a_device,
             image_a_laba.get(),
             vcsmc::kLabaBufferSizeBytes,
             cudaMemcpyHostToDevice);
  cudaMemcpy(image_b_device,
             image_b_laba.get(),
             vcsmc::kLabaBufferSizeBytes,
             cudaMemcpyHostToDevice);

  dim3 dim_grid(vcsmc::kTargetFrameWidthPixels / 32,
                vcsmc::kFrameHeightPixels / 32);
  dim3 dim_block(32, 32);

  float* nl_a_device;
  cudaMalloc(&nl_a_device, vcsmc::kLBufferSizeBytes);
  float* nl_b_device;
  cudaMalloc(&nl_b_device, vcsmc::kLBufferSizeBytes);
  vcsmc::LabaToNormalizedL<<<dim_grid, dim_block>>>(
      image_a_device, nl_a_device);
  vcsmc::LabaToNormalizedL<<<dim_grid, dim_block>>>(
      image_b_device, nl_b_device);

  float* mean_a_device;
  cudaMalloc(&mean_a_device, vcsmc::kLBufferSizeBytes);
  float* mean_b_device;
  cudaMalloc(&mean_b_device, vcsmc::kLBufferSizeBytes);
  vcsmc::ComputeMean<<<dim_grid, dim_block>>>(nl_a_device, mean_a_device);
  vcsmc::ComputeMean<<<dim_grid, dim_block>>>(nl_b_device, mean_b_device);

  float* variance_a_device;
  cudaMalloc(&variance_a_device, vcsmc::kLBufferSizeBytes);
  float* variance_b_device;
  cudaMalloc(&variance_b_device, vcsmc::kLBufferSizeBytes);
  vcsmc::ComputeVariance<<<dim_grid, dim_block>>>(
      nl_a_device, mean_a_device, variance_a_device);
  vcsmc::ComputeVariance<<<dim_grid, dim_block>>>(
      nl_b_device, mean_b_device, variance_b_device);

  float* covariance_device;
  cudaMalloc(&covariance_device, vcsmc::kLBufferSizeBytes);
  vcsmc::ComputeCovariance<<<dim_grid, dim_block>>>(nl_a_device,
                                                    mean_a_device,
                                                    nl_b_device,
                                                    mean_b_device,
                                                    covariance_device);

  float* ssim_device;
  cudaMalloc(&ssim_device, vcsmc::kLBufferSizeBytes);
  vcsmc::ComputeSSIM<<<dim_grid, dim_block>>>(mean_a_device,
                                              variance_a_device,
                                              mean_b_device,
                                              variance_b_device,
                                              covariance_device,
                                              ssim_device);

  if (FLAGS_map_image != "") {
    std::unique_ptr<float> ssim_map(new float[vcsmc::kLBufferSize]);
    cudaMemcpy(ssim_map.get(), ssim_device, vcsmc::kLBufferSizeBytes,
        cudaMemcpyDeviceToHost);
    vcsmc::GrayMap::Save(ssim_map.get(), vcsmc::kTargetFrameWidthPixels,
        vcsmc::kFrameHeightPixels, FLAGS_map_image);
  }

  float* block_sums_device;
  cudaMalloc(&block_sums_device, vcsmc::kBlockSumBufferSizeBytes);
  vcsmc::ComputeBlockSum<<<dim_grid, dim_block, 1024 * sizeof(float)>>>(
      ssim_device, block_sums_device);

  std::unique_ptr<float> block_sums(new float[vcsmc::kBlockSumBufferSize]);
  cudaMemcpy(block_sums.get(), block_sums_device,
      vcsmc::kBlockSumBufferSizeBytes, cudaMemcpyDeviceToHost);

  float sum = 0.0;
  for (size_t i = 0; i < vcsmc::kBlockSumBufferSize; ++i) {
    sum += block_sums.get()[i];
  }

  float mssim = sum / (static_cast<float>(vcsmc::kTargetFrameWidthPixels *
      vcsmc::kFrameHeightPixels));

  printf("mssim: %f, score: %f\n", mssim, 1.0f - mssim);

  cudaFree(block_sums_device);
  cudaFree(ssim_device);
  cudaFree(covariance_device);
  cudaFree(variance_b_device);
  cudaFree(variance_a_device);
  cudaFree(mean_b_device);
  cudaFree(mean_a_device);
  cudaFree(nl_b_device);
  cudaFree(nl_a_device);
  cudaFree(image_b_device);
  cudaFree(image_a_device);
}
