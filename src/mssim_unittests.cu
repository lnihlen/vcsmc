#include "mssim.h"

#include <cstring>

#include "gtest/gtest.h"

namespace {

std::unique_ptr<float> MakeInputBlock() {
  std::unique_ptr<float> nyuv_input(new float[vcsmc::kNyuvBufferSize]);
  float* nyuv_ptr = nyuv_input.get();

  // Zero nyuv buffer to include padding.
  std::memset(nyuv_ptr, 0, vcsmc::kNyuvBufferSizeBytes);

  // Populate each block of pixels starting from upper left hand corner with
  // (i / kWindowSize, i / kWindowSize + 1000, -(i / kWindowSize + 1000))
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      float val = static_cast<float>(
          ((i / vcsmc::kWindowSize) *
          (vcsmc::kTargetFrameWidthPixels / vcsmc::kWindowSize)) +
          (j / vcsmc::kWindowSize));
      *nyuv_ptr = val;
      ++nyuv_ptr;
      *nyuv_ptr = val + 1000.0f;
      ++nyuv_ptr;
      *nyuv_ptr = -(val + 1000.0f);
      ++nyuv_ptr;
      *nyuv_ptr = 1.0f;
      ++nyuv_ptr;
    }
    // Skip padding on right.
    nyuv_ptr +=  4 * vcsmc::kWindowSize;
  }

  return nyuv_input;
}


}  // namespace

namespace vcsmc {

TEST(MssimTest, LocalMeanTest) {
  std::unique_ptr<float> nyuv_input = MakeInputBlock();

  // Copy nyuv buffer to device, compute mean, copy it back.
  float4* nyuv_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nyuv_device, vcsmc::kNyuvBufferSizeBytes));
  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kNyuvBufferSizeBytes));
  dim3 image_dim_block(16, 16);
  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(nyuv_device, nyuv_input.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  vcsmc::ComputeLocalMean<<<image_dim_block, image_dim_grid>>>(
      nyuv_device, mean_device);
  std::unique_ptr<float> mean_output(new float[vcsmc::kNyuvBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_output.get(), mean_device,
      vcsmc::kNyuvBufferSizeBytes, cudaMemcpyDeviceToHost));

  ASSERT_EQ(cudaSuccess, cudaFree(nyuv_device));
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
