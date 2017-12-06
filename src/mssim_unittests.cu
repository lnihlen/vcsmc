#include "mssim.h"

#include <cstring>

#include "gtest/gtest.h"

namespace vcsmc {

TEST(MssimTest, LocalMeanTest) {
  std::unique_ptr<float> lab_input(new float[vcsmc::kLabBufferSize]);
  float* lab_ptr = lab_input.get();

  // Zero lab buffer to include padding.
  std::memset(lab_ptr, 0, kLabBufferSizeBytes);

  // Populate each block of pixels starting from upper left hand corner with
  // (i / kWindowSize, i / kWindowSize + 1000, -(i / kWindowSize + 1000))
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      float val = static_cast<float>(
          ((i / kWindowSize) *
           (vcsmc::kTargetFrameWidthPixels / kWindowSize)) + (j / kWindowSize));
      *lab_ptr = val;
      ++lab_ptr;
      *lab_ptr = val + 1000.0f;
      ++lab_ptr;
      *lab_ptr = -(val + 1000.0f);
      ++lab_ptr;
      *lab_ptr = 1.0f;
      ++lab_ptr;
    }
    // Skip padding on right.
    lab_ptr +=  4 * kWindowSize;
  }

  // Copy lab buffer to device, compute mean, copy it back.
  float4* lab_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&lab_device, vcsmc::kLabBufferSizeBytes));
  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLabBufferSizeBytes));
  dim3 image_dim_block(16, 16);
  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(lab_device, lab_input.get(),
        vcsmc::kLabBufferSizeBytes, cudaMemcpyHostToDevice));
  vcsmc::ComputeLocalMean<<<image_dim_block, image_dim_grid>>>(
      lab_device, mean_device);
  std::unique_ptr<float> mean_output(new float[vcsmc::kLabBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_output.get(), mean_device,
      vcsmc::kLabBufferSizeBytes, cudaMemcpyDeviceToHost));

  ASSERT_EQ(cudaSuccess, cudaFree(lab_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));

  // Validate contents of mean buffer.
  const float* mean = mean_output.get();
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    // Last kWindowSize rows have same values due to absent higher-value lower
    // image rows.
    size_t y = i > vcsmc::kFrameHeightPixels - kWindowSize ?
        vcsmc::kFrameHeightPixels - kWindowSize : i;
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      // Last kWindowSize pixels in row should hold constant value, as we don't
      // sample next row.
      size_t x = j > vcsmc::kTargetFrameWidthPixels - kWindowSize ?
          vcsmc::kTargetFrameWidthPixels - kWindowSize : j;
      float target = static_cast<float>(
          x + (y * vcsmc::kTargetFrameWidthPixels / kWindowSize))
          / static_cast<float>(kWindowSize);
      ASSERT_EQ(target, *mean);
      ++mean;
      ASSERT_EQ(target + 1000.0f, *mean);
      ++mean;
      ASSERT_EQ(-(target + 1000.0f), *mean);
      ++mean;
      ASSERT_EQ(1.0f, *mean);
      ++mean;
    }
    // Validate padding at end of row is all zeros, including a zero in the
    // fourth element which indicates masked data.
    for (size_t j = 0; j < kWindowSize * 4; ++j) {
      ASSERT_EQ(0.0f, *mean);
      ++mean;
    }
  }
  // Validate padding rows at bottom of the image are also all zeros.
  for (size_t i = 0;
       i < (vcsmc::kTargetFrameWidthPixels + kWindowSize) * kWindowSize * 4;
       ++i) {
    ASSERT_EQ(0.0f, *mean);
    ++mean;
  }
}

}  // namespace vcsmc
