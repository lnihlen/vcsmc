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

std::unique_ptr<float> ComputePaddedMean(const float* input) {
  std::unique_ptr<float> mean(new float[vcsmc::kNyuvBufferSize]);
  float* mean_ptr = mean.get();
  std::memset(mean_ptr, 0, vcsmc::kNyuvBufferSizeBytes);

  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      uint32 mean_offset =
          ((i * (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize)) + j) * 4;
      float sum_y = 0.0f;
      float sum_u = 0.0f;
      float sum_v = 0.0f;
      float n = 0.0f;
      for (size_t y = 0; y < vcsmc::kWindowSize; ++y) {
        if (i + y >= vcsmc::kFrameHeightPixels) continue;
        uint32 offset = mean_offset +
            (y * (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize)) * 4;
        for (size_t x = 0; x < vcsmc::kWindowSize; ++x) {
          if (j + x >= vcsmc::kTargetFrameWidthPixels) continue;
          sum_y += input[offset];
          sum_u += input[offset + 1];
          sum_v += input[offset + 2];
          n += input[offset + 3];
          offset += 4;
        }
      }
      mean_ptr[mean_offset] = sum_y / n;
      mean_ptr[mean_offset + 1] = sum_u / n;
      mean_ptr[mean_offset + 2] = sum_v / n;
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
      float sum_y = 0.0f;
      float sum_u = 0.0f;
      float sum_v = 0.0f;
      float mean_y = mean[stddevsq_offset];
      float mean_u = mean[stddevsq_offset + 1];
      float mean_v = mean[stddevsq_offset + 2];
      float n = 0.0f;
      for (size_t y = 0; y < vcsmc::kWindowSize; ++y) {
        uint32 offset = stddevsq_offset +
            ((y * (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize)) * 4);
        for (size_t x = 0; x < vcsmc::kWindowSize; ++x) {
          float del_y = input[offset] - mean_y;
          sum_y += del_y * del_y;
          float del_u = input[offset + 1] - mean_u;
          sum_u += del_u * del_u;
          float del_v = input[offset + 2] - mean_v;
          sum_v += del_v * del_v;
          n += input[offset + 3];
          offset += 4;
        }
      }
      float n_minus_one = std::max(1.0f, n - 1.0f);
      stddevsq_ptr[stddevsq_offset] = sum_y / n_minus_one;
      stddevsq_ptr[stddevsq_offset + 1] = sum_u / n_minus_one;
      stddevsq_ptr[stddevsq_offset + 2] = sum_v / n_minus_one;
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

int ulp_delta(float a, float b) {
  int* a_int = reinterpret_cast<int*>(&a);
  int* b_int = reinterpret_cast<int*>(&b);
  return std::abs(*a_int - *b_int);
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
}


TEST(MssimTest, MeanTest) {
  std::unique_ptr<float> nyuv_input = MakePaddedTestInput();

  // Copy nyuv buffer to device, compute mean, copy it back.
  float4* nyuv_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nyuv_device, vcsmc::kNyuvBufferSizeBytes));
  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kNyuvBufferSizeBytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nyuv_device, nyuv_input.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));

  dim3 image_dim_block(16, 16);
  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  vcsmc::ComputeLocalMean<<<image_dim_block, image_dim_grid>>>(
      nyuv_device, mean_device);

  std::unique_ptr<float> mean_output(new float[vcsmc::kNyuvBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_output.get(), mean_device,
      vcsmc::kNyuvBufferSizeBytes, cudaMemcpyDeviceToHost));

  std::unique_ptr<float> nyuv_mean_generated = MakePaddedTestMean();

  EXPECT_EQ(0, std::memcmp(mean_output.get(),
                           nyuv_mean_generated.get(),
                           vcsmc::kNyuvBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaFree(nyuv_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
}

TEST(MssimTest, StandardDeviationSquaredTest) {
  std::unique_ptr<float> nyuv_input = MakePaddedTestInput();
  std::unique_ptr<float> nyuv_mean = MakePaddedTestMean();

  float4* nyuv_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nyuv_device, vcsmc::kNyuvBufferSizeBytes));
  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kNyuvBufferSizeBytes));
  float4* stddevsq_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&stddevsq_device,
      vcsmc::kNyuvBufferSizeBytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nyuv_device, nyuv_input.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_device, nyuv_mean.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));

  dim3 image_dim_block(16, 16);
  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  vcsmc::ComputeLocalStdDevSquared<<<image_dim_block, image_dim_grid>>>(
      nyuv_device, mean_device, stddevsq_device);

  std::unique_ptr<float> stddevsq_output(new float[vcsmc::kNyuvBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(stddevsq_output.get(), stddevsq_device,
      vcsmc::kNyuvBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(stddevsq_output.get());

  std::unique_ptr<float> nyuv_stddevsq_computed =
      ComputePaddedStandardDeviationSquared(nyuv_input.get(), nyuv_mean.get());

  for (size_t i = 0; i < vcsmc::kNyuvBufferSize; i += 4) {
    EXPECT_LE(ulp_delta(nyuv_stddevsq_computed.get()[i],
                        stddevsq_output.get()[i]), 32);
    EXPECT_LE(ulp_delta(nyuv_stddevsq_computed.get()[i + 1],
                        stddevsq_output.get()[i + 1]), 32);
    EXPECT_LE(ulp_delta(nyuv_stddevsq_computed.get()[i + 2],
                        stddevsq_output.get()[i + 2]), 32);
    EXPECT_EQ(nyuv_stddevsq_computed.get()[i + 3],
              stddevsq_output.get()[i + 3]);
  }

  ASSERT_EQ(cudaSuccess, cudaFree(nyuv_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
  ASSERT_EQ(cudaSuccess, cudaFree(stddevsq_device));
}

}  // namespace vcsmc
