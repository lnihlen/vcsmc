#include "mssim.h"

#include <cstring>
#include <cmath>

#include "gtest/gtest.h"

namespace {

std::unique_ptr<float> MakePaddedTestInput() {
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
      *nyuv_ptr = val + 0.5f;
      ++nyuv_ptr;
      *nyuv_ptr = -(val + 0.5f);
      ++nyuv_ptr;
      *nyuv_ptr = 1.0f;
      ++nyuv_ptr;
    }
    // Skip padding on right.
    nyuv_ptr +=  4 * vcsmc::kWindowSize;
  }

  return nyuv_input;
}

std::unique_ptr<float> MakePaddedTestMean() {
  std::unique_ptr<float> mean(new float[vcsmc::kNyuvBufferSize]);
  float* mean_ptr = mean.get();
  std::memset(mean_ptr, 0, vcsmc::kNyuvBufferSizeBytes);

  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    // Last kWindowSize rows have same values due to absent higher-value lower
    // image rows.
    size_t y = i > vcsmc::kFrameHeightPixels - vcsmc::kWindowSize ?
        vcsmc::kFrameHeightPixels - vcsmc::kWindowSize : i;
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      // Last kWindowSize pixels in row should hold constant value, as we don't
      // sample next row.
      size_t x = j > vcsmc::kTargetFrameWidthPixels - vcsmc::kWindowSize ?
          vcsmc::kTargetFrameWidthPixels - vcsmc::kWindowSize : j;
      float target = static_cast<float>(
          x + (y * vcsmc::kTargetFrameWidthPixels / vcsmc::kWindowSize))
          / static_cast<float>(vcsmc::kWindowSize);
      *mean_ptr = target;
      ++mean_ptr;
      *mean_ptr = target + 0.5f;
      ++mean_ptr;
      *mean_ptr = -(target + 0.5f);
      ++mean_ptr;
      *mean_ptr = 1.0f;
      ++mean_ptr;
    }

    // Skip padding at end of row.
    mean_ptr += vcsmc::kWindowSize * 4;
  }

  return mean;
}

/*
    0        7       12       15       16       15       12        7
11200    11207    11212    11215    11216    11215    11212    11207
19200    19207    19212    19215    19216    19215    19212    19207
24000    24007    24012    24015    24016    24015    24012    24007
25600    25607    25612    25615    25616    25615    25612    25607
24000    24007    24012    24015    24016    24015    24012    24007
19200    19207    19212    19215    19216    19215    19212    19207
11200    11207    11212    11215    11216    11215    11212    11207
*/
std::unique_ptr<float> MakePaddedTestStandardDeviationSquared() {
  const float[] collumn_values = { 0.0f, 7.0f, 12.0f, 15.0f, 16.0f, 15.0f, 12.0f, 7.0f };
  const float[] row_values = { 0.0f, 11200.0f, 19200.0f, 24000.0f, 25600.0f, 24000.0f, 19200.0f, 11200.0f };

}


std::unique_ptr<float> ComputePaddedMean(const float* input) {
  std::unique_ptr<float> mean(new float[vcsmc::kNyuvBufferSize]);
  float* mean_ptr = mean.get();
  std::memset(mean_ptr, 0, vcsmc::kNyuvBufferSizeBytes);

  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      uint32 mean_offset =
          ((i * (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize)) + j) * 4;
      float sum_Y = 0.0f;
      float sum_U = 0.0f;
      float sum_V = 0.0f;
      uint32 n = 0;
      for (size_t y = 0; y < vcsmc::kWindowSize; ++y) {
        if (i + y >= vcsmc::kFrameHeightPixels) continue;
        uint32 offset = mean_offset +
            (y * (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize)) * 4;
        for (size_t x = 0; x < vcsmc::kWindowSize; ++x) {
          if (j + x >= vcsmc::kTargetFrameWidthPixels) continue;
          sum_Y += input[offset];
          sum_U += input[offset + 1];
          sum_V += input[offset + 2];
          offset += 4;
          ++n;
        }
      }
      float n_float = static_cast<float>(n);
      mean_ptr[mean_offset] = sum_Y / n_float;
      mean_ptr[mean_offset + 1] = sum_U / n_float;
      mean_ptr[mean_offset + 2] = sum_V / n_float;
      mean_ptr[mean_offset + 3] = 1.0f;
    }
  }

  return mean;
}

std::unique_ptr<float> ComputePaddedStandardDeviationSquared(
    const float* input,
    const float* mean) {
  std::unique_ptr<float> stddevsq(new float[vcsmc::kNyuvBufferSize]);
  float* stddevsq_ptr = stddevsq.get();
  std::memset(stddevsq_ptr, 0, vcsmc::kNyuvBufferSizeBytes);

  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      uint32 stddevsq_offset =
          ((i * (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize)) + j) * 4;
      float mean_Y_at_offset = mean[stddevsq_offset];
      float mean_U_at_offset = mean[stddevsq_offset + 1];
      float mean_V_at_offset = mean[stddevsq_offset + 2];
      float sum_Y = 0.0f;
      float sum_U = 0.0f;
      float sum_V = 0.0f;
      uint32 n = 0;
      for (size_t y = 0; y < vcsmc::kWindowSize; ++y) {
        if (i + y >= vcsmc::kFrameHeightPixels) continue;
        uint32 offset = stddevsq_offset +
            (y * (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize)) * 4;
        for (size_t x = 0; x < vcsmc::kWindowSize; ++x) {
          if (j + x >= vcsmc::kTargetFrameWidthPixels) continue;
          sum_Y += std::pow(input[offset] - mean_Y_at_offset, 2.0f);
          sum_U += std::pow(input[offset + 1] - mean_U_at_offset, 2.0f);
          sum_V += std::pow(input[offset + 2] - mean_V_at_offset, 2.0f);
          offset += 4;
          ++n;
        }
      }
      float n_minus_one = std::min(1.0f, static_cast<float>(n) - 1.0f);
      stddevsq_ptr[stddevsq_offset] = sum_Y / n_minus_one;
      stddevsq_ptr[stddevsq_offset + 1] = sum_U / n_minus_one;
      stddevsq_ptr[stddevsq_offset + 2] = sum_V / n_minus_one;
      stddevsq_ptr[stddevsq_offset + 3] = 1.0f;
    }
  }

  return stddevsq;
}

// Given a padded float array |input| ensure that the fourth element of each
// float4 vector is a 1.0f in the area of valid data, and 0.0f in the area of
// padding.
void ValidatePaddingWeights(const float* input) {
  uint32 offset = 0;
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      EXPECT_EQ(1.0f, input[offset + 3]);
      offset += 4;
    }

    for (size_t j = 0; j < vcsmc::kWindowSize; ++j) {
      EXPECT_EQ(0.0f, input[offset]);
      EXPECT_EQ(0.0f, input[offset + 1]);
      EXPECT_EQ(0.0f, input[offset + 2]);
      EXPECT_EQ(0.0f, input[offset + 3]);
      offset += 4;
    }
  }

  for (size_t i = 0;
       i < 4 * vcsmc::kWindowSize *
       (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize); ++i) {
    EXPECT_EQ(0.0f, input[offset]);
    ++offset;
  }

  // Should have checked every float in the buffer.
  ASSERT_EQ(vcsmc::kNyuvBufferSize, offset);
}

void PrintBlock(const float* p, size_t n) {
  for (size_t i = 0; i < 8; ++i) {
    for (size_t j = 0; j < n * 32; j += 32) {
      printf("%12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f",
        p[j], p[j+4], p[j+8], p[j+12], p[j+16], p[j+20], p[j+24], p[j+28]);
    }
  printf("\n");
  p += 4 * (vcsmc::kWindowSize + vcsmc::kTargetFrameWidthPixels);
  }
}

}  // namespace

namespace vcsmc {

// Sanity-check our own computed functions against our generated synthetic
// inputs.
TEST(MssimTest, ComputedStatisticsTest) {
  std::unique_ptr<float> nyuv_input = MakePaddedTestInput();
  ValidatePaddingWeights(nyuv_input.get());

  std::unique_ptr<float> nyuv_mean_generated = MakePaddedTestMean();
  ValidatePaddingWeights(nyuv_mean_generated.get());

  std::unique_ptr<float> nyuv_mean_computed =
    ComputePaddedMean(nyuv_input.get());
  ValidatePaddingWeights(nyuv_mean_computed.get());

  ASSERT_EQ(0, std::memcmp(nyuv_mean_generated.get(),
                           nyuv_mean_computed.get(),
                           vcsmc::kNyuvBufferSizeBytes));

  std::unique_ptr<float> nyuv_stddevsq_computed =
      ComputePaddedStandardDeviationSquared(nyuv_input.get(),
                                            nyuv_mean_generated.get());
  ValidatePaddingWeights(nyuv_stddevsq_computed.get());

  printf("nyuv_input Y:\n");
  PrintBlock(nyuv_input.get(), 2);

  printf("nyuv_input U:\n");
  PrintBlock(nyuv_input.get() + 1, 2);

  printf("nyuv_input U:\n");
  PrintBlock(nyuv_input.get() + 2, 2);

  printf("\ncomputed mean Y:\n");
  PrintBlock(nyuv_mean_computed.get(), 2);
  printf("computed mean U:\n");
  PrintBlock(nyuv_mean_computed.get() + 1, 2);
  printf("computed mean V:\n");
  PrintBlock(nyuv_mean_computed.get() + 2, 2);

  printf("\nstddev Y:\n");
  PrintBlock(nyuv_stddevsq_computed.get(), 2);
  PrintBlock(nyuv_stddevsq_computed.get() + ((vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize) * 4 * vcsmc::kWindowSize), 2);

  printf("stddev U:\n");
  PrintBlock(nyuv_stddevsq_computed.get() + 1, 2);
  PrintBlock(nyuv_stddevsq_computed.get() + ((vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize) * 4 * vcsmc::kWindowSize) + 1, 2);

  printf("stddev V:\n");
  PrintBlock(nyuv_stddevsq_computed.get() + 2, 2);
  PrintBlock(nyuv_stddevsq_computed.get() + ((vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize) * 4 * vcsmc::kWindowSize) + 2, 2);
  printf("\n");
}


TEST(MssimTest, MeanTest) {
  std::unique_ptr<float> nyuv_input = MakePaddedTestInput();

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

  std::unique_ptr<float> nyuv_mean_generated = MakePaddedTestMean();

  ASSERT_EQ(0, std::memcmp(mean_output.get(),
                           nyuv_mean_generated.get(),
                           vcsmc::kNyuvBufferSizeBytes));
}

}  // namespace vcsmc
