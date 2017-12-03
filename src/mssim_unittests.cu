#include "mssim.h"

#include "gtest/gtest.h"

namespace vcsmc {

const size_t kBlockSize = 8;

TEST(MssimTest, LocalMeanTest) {
  std::unique_ptr<float> lab_input(new float[vcsmc::kFrameSizeBytes * 3]);
  float* lab_ptr = lab_input.get();

  // Populate each block of pixels starting from upper left hand corner with
  // (i / kBlockSize, i / kBlockSize + 1000, -(i / kBlockSize + 1000))
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels / kBlockSize; ++i) {
    for (size_t j = 0; j < kBlockSize; ++j) {
      for (size_t k = 0; k < vcsmc::kTargetFrameWidthPixels / kBlockSize; ++k) {
        float val = static_cast<float>(
            (i * (vcsmc::kTargetFrameWidthPixels / kBlockSize)) + k);
        for (size_t l = 0; l < kBlockSize; ++l) {
          *lab_ptr = val;
          ++lab_ptr;
          *lab_ptr = val + 1000;
          ++lab_ptr;
          *lab_ptr = -(val + 1000);
          ++lab_ptr;
        }
      }
    }
  }

  // Copy lab buffer to device, compute mean, copy it back.
  float3* lab_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&lab_device, vcsmc::kLabBufferSizeBytes));
  float3* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLabBufferSizeBytes));
  dim3 image_dim_block(16, 16);
  dim3 image_dim_grid(vcsmc::kTargetFrameWidthPixels / 16,
                      vcsmc::kFrameHeightPixels / 16);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(lab_device, lab_input.get(),
        vcsmc::kLabBufferSizeBytes, cudaMemcpyHostToDevice));
  vcsmc::ComputeLocalMean<<<image_dim_block, image_dim_grid>>>(
      lab_device, mean_device);
  std::unique_ptr<float> mean_output(new float[vcsmc::kFrameSizeBytes * 3]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_output.get(), mean_device,
      vcsmc::kLabBufferSizeBytes, cudaMemcpyDeviceToHost));

  ASSERT_EQ(cudaSuccess, cudaFree(lab_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));

  // Validate contents of mean buffer.
  float* mean = mean_output.get();
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    // Last kBlockSize rows have same values due to absent higher-value lower
    // image rows.
    size_t y = i > vcsmc::kFrameHeightPixels - kBlockSize ?
        vcsmc::kFrameHeightPixels - kBlockSize : i;
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      // Last kBlockSize pixels in row should hold constant value, as we don't
      // sample next row.
      size_t x = j > vcsmc::kTargetFrameWidthPixels - kBlockSize ?
          vcsmc::kTargetFrameWidthPixels - kBlockSize : j;
      float target = static_cast<float>(
          x + (y * vcsmc::kTargetFrameWidthPixels / kBlockSize))
          / static_cast<float>(kBlockSize);
      ASSERT_EQ(target, *mean);
      ++mean;
      ASSERT_EQ(target + 1000.0f, *mean);
      ++mean;
      ASSERT_EQ(-(target + 1000.0f), *mean);
      ++mean;
    }
  }
}

}  // namespace vcsmc
