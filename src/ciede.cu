// ciede - given two images generates an error map and reports mean normalized
// ciede distance for those two images.

#include <gflags/gflags.h>

#include "cuda.h"
#include "cuda_runtime.h"

#include "ciede_2k.h"
#include "color.h"
#include "cuda_utils.h"
#include "gray_map.h"
#include "image.h"
#include "image_file.h"
#include "mssim.h"

DEFINE_string(image_a, "", "Required - path to first image for comparison.");
DEFINE_string(image_b, "", "Required - path to second image for comparison.");
DEFINE_string(map_image, "", "Optional - path for grayscale output image for "
    "ciede map.");

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

  float* ciede_device;
  cudaMalloc(&ciede_device, vcsmc::kLBufferSizeBytes);

  dim3 dim_grid(vcsmc::kTargetFrameWidthPixels / 32,
                vcsmc::kFrameHeightPixels / 32);
  dim3 dim_block(32, 32);
  vcsmc::Ciede2k<<<dim_grid, dim_block>>>(image_a_device,
                                          image_b_device,
                                          ciede_device);
  if (FLAGS_map_image != "") {
    std::unique_ptr<float> ciede_map(new float[vcsmc::kLBufferSize]);
    cudaMemcpy(ciede_map.get(), ciede_device, vcsmc::kLBufferSizeBytes,
        cudaMemcpyDeviceToHost);
    vcsmc::GrayMap::Save(ciede_map.get(), vcsmc::kTargetFrameWidthPixels,
        vcsmc::kFrameHeightPixels, FLAGS_map_image);
  }

  float* block_sums_device;
  cudaMalloc(&block_sums_device, vcsmc::kBlockSumBufferSizeBytes);
  vcsmc::ComputeBlockSum<<<dim_grid, dim_block, 1024 * sizeof(float)>>>(
      ciede_device, block_sums_device);

  std::unique_ptr<float> block_sums(new float[vcsmc::kBlockSumBufferSize]);
  cudaMemcpy(block_sums.get(), block_sums_device,
      vcsmc::kBlockSumBufferSizeBytes, cudaMemcpyDeviceToHost);

  float sum = 0.0;
  for (size_t i = 0; i < vcsmc::kBlockSumBufferSize; ++i) {
    sum += block_sums.get()[i];
  }

  float mean_ciede = sum / (static_cast<float>(vcsmc::kTargetFrameWidthPixels)
      * static_cast<float>(vcsmc::kFrameHeightPixels));

  printf("mean ciede: %f\n", mean_ciede);

  cudaFree(block_sums_device);
  cudaFree(ciede_device);
  cudaFree(image_b_device);
  cudaFree(image_a_device);
}
