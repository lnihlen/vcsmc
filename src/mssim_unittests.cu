#include "mssim.h"

#include <cstring>
#include <cmath>

#include "gtest/gtest.h"

namespace {

std::unique_ptr<float> MakePaddedTestInput(float slope = 1.0f,
                                           float intercept = 0.0f) {
  std::unique_ptr<float> nyuv_input(new float[vcsmc::kNyuvBufferSize]);
  float* nyuv_ptr = nyuv_input.get();

  // Zero nyuv buffer to include padding.
  std::memset(nyuv_ptr, 0, vcsmc::kNyuvBufferSizeBytes);

  // Populate each block of pixels starting from upper left hand corner with
  // (i / kWindowSize, i / kWindowSize + 1000, -(i / kWindowSize + 1000))
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      float val = (static_cast<float>(
          ((i / vcsmc::kWindowSize) *
          (vcsmc::kTargetFrameWidthPixels / vcsmc::kWindowSize)) +
          (j / vcsmc::kWindowSize)) * slope) + intercept;
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

// Assumes MakePaddedTestInput was called with default arguments.
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
          float pad = input[offset + 3];
          float del_y = input[offset] - mean_y;
          sum_y += del_y * del_y * pad;
          float del_u = input[offset + 1] - mean_u;
          sum_u += del_u * del_u * pad;
          float del_v = input[offset + 2] - mean_v;
          sum_v += del_v * del_v * pad;
          n += pad;
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

// Returns distance between |a| and |b| in ulps.
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

  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  dim3 image_dim_block(16, 16);
  vcsmc::ComputeLocalMean<<<image_dim_grid, image_dim_block>>>(
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

  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  dim3 image_dim_block(16, 16);
  vcsmc::ComputeLocalStdDevSquared<<<image_dim_grid, image_dim_block>>>(
      nyuv_device, mean_device, stddevsq_device);

  std::unique_ptr<float> stddevsq_output(new float[vcsmc::kNyuvBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(stddevsq_output.get(), stddevsq_device,
      vcsmc::kNyuvBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(stddevsq_output.get());

  std::unique_ptr<float> nyuv_stddevsq_computed =
      ComputePaddedStandardDeviationSquared(nyuv_input.get(), nyuv_mean.get());

  EXPECT_EQ(0, std::memcmp(nyuv_stddevsq_computed.get(),
                           stddevsq_output.get(),
                           vcsmc::kNyuvBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaFree(nyuv_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
  ASSERT_EQ(cudaSuccess, cudaFree(stddevsq_device));
}

TEST(MssimTest, CovarianceTest) {
  std::unique_ptr<float> nyuv_input = MakePaddedTestInput();
  std::unique_ptr<float> nyuv_mean = MakePaddedTestMean();

  float4* nyuv_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nyuv_device, vcsmc::kNyuvBufferSizeBytes));
  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kNyuvBufferSizeBytes));
  float4* covariance_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&covariance_device,
                                    vcsmc::kNyuvBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(nyuv_device, nyuv_input.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_device, nyuv_mean.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));

  // Covariance of variable with itself should be equal to variance, or
  // standard deviation squared.
  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  dim3 image_dim_block(16, 16);
  vcsmc::ComputeLocalCovariance<<<image_dim_grid, image_dim_block>>>(
      nyuv_device, mean_device, nyuv_device, mean_device, covariance_device);

  std::unique_ptr<float> covariance(new float[vcsmc::kNyuvBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kNyuvBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(covariance.get());

  std::unique_ptr<float> variance_computed =
      ComputePaddedStandardDeviationSquared(nyuv_input.get(), nyuv_mean.get());

  EXPECT_EQ(0, std::memcmp(variance_computed.get(),
                           covariance.get(),
                           vcsmc::kNyuvBufferSizeBytes));

  // cov(X + a, Y + b) = cov(X, Y) for constants a,b, random variables X, Y.
  std::unique_ptr<float> nyuv_plus_a = MakePaddedTestInput(1.0f, 2.0f);
  std::unique_ptr<float> nyuv_mean_a = ComputePaddedMean(nyuv_plus_a.get());
  std::unique_ptr<float> nyuv_plus_b = MakePaddedTestInput(1.0f, -4.0f);
  std::unique_ptr<float> nyuv_mean_b = ComputePaddedMean(nyuv_plus_b.get());

  float4* nyuv_a_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nyuv_a_device,
                                    vcsmc::kNyuvBufferSizeBytes));
  float4* mean_a_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_a_device,
                                    vcsmc::kNyuvBufferSizeBytes));
  float4* nyuv_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nyuv_b_device,
                                    vcsmc::kNyuvBufferSizeBytes));
  float4* mean_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_b_device,
                                    vcsmc::kNyuvBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(nyuv_a_device, nyuv_plus_a.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_a_device, nyuv_mean_a.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nyuv_b_device, nyuv_plus_b.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_b_device, nyuv_mean_b.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));

  vcsmc::ComputeLocalCovariance<<<image_dim_grid, image_dim_block>>>(
      nyuv_a_device, mean_a_device, nyuv_b_device, mean_b_device,
      covariance_device);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kNyuvBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(covariance.get());

  EXPECT_EQ(0, std::memcmp(variance_computed.get(),
                           covariance.get(),
                           vcsmc::kNyuvBufferSizeBytes));

  // cov(aX, bY) = ab cov(X,Y) for constants a,b, random variables X, Y.
  std::unique_ptr<float> nyuv_times_a = MakePaddedTestInput(-5.0f, 0.0f);
  nyuv_mean_a = ComputePaddedMean(nyuv_times_a.get());
  std::unique_ptr<float> nyuv_times_b = MakePaddedTestInput(3.0f, 0.0f);
  nyuv_mean_b = ComputePaddedMean(nyuv_times_b.get());

  ASSERT_EQ(cudaSuccess, cudaMemcpy(nyuv_a_device, nyuv_times_a.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_a_device, nyuv_mean_a.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nyuv_b_device, nyuv_times_b.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_b_device, nyuv_mean_b.get(),
        vcsmc::kNyuvBufferSizeBytes, cudaMemcpyHostToDevice));

  vcsmc::ComputeLocalCovariance<<<image_dim_grid, image_dim_block>>>(
      nyuv_a_device, mean_a_device, nyuv_b_device, mean_b_device,
      covariance_device);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kNyuvBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(covariance.get());

  for (size_t i = 0; i < vcsmc::kNyuvBufferSize; i += 4) {
    EXPECT_GT(16, ulp_delta(-15.0f * variance_computed.get()[i],
                           covariance.get()[i]));
    EXPECT_GT(16, ulp_delta(-15.0f * variance_computed.get()[i + 1],
                           covariance.get()[i + 1]));
    EXPECT_GT(16, ulp_delta(-15.0f * variance_computed.get()[i + 2],
                           covariance.get()[i + 2]));
    EXPECT_EQ(variance_computed.get()[i + 3],
              covariance.get()[i + 3]);
  }

  ASSERT_EQ(cudaSuccess, cudaFree(nyuv_a_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_a_device));
  ASSERT_EQ(cudaSuccess, cudaFree(nyuv_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(covariance_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
  ASSERT_EQ(cudaSuccess, cudaFree(nyuv_device));
}

TEST(MssimTest, BlockSumTest) {
  std::unique_ptr<float> zeros(new float[vcsmc::kFrameSizeBytes]);
  std::memset(zeros.get(), 0, sizeof(float) * vcsmc::kFrameSizeBytes);

  float* input_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&input_device, vcsmc::kFrameSizeBytes *
      sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(input_device, zeros.get(),
      vcsmc::kFrameSizeBytes * sizeof(float), cudaMemcpyHostToDevice));

  float* results_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&results_device, 60 * sizeof(float)));

  dim3 grid_dim(10, 6);
  dim3 block_dim(32, 32);
  vcsmc::ComputeBlockSum<<<grid_dim, block_dim, 1024 * sizeof(float)>>>(
      input_device, results_device);

  std::unique_ptr<float> results(new float[60]);
  // Paint the results with 1.0 so we can detect that we actually got zeros back
  // from the device.
  for (size_t i = 0; i < 60; ++i) {
    results.get()[i] = 1.0f;
  }

  ASSERT_EQ(cudaSuccess, cudaMemcpy(results.get(), results_device,
      60 * sizeof(float), cudaMemcpyDeviceToHost));

  // Sums of all zeros should be zero.
  for (size_t i = 0; i < 60; ++i) {
    EXPECT_EQ(0.0f, results.get()[i]);
  }

  // Build blocks that contain values from -511 to +511, plus the index
  // of the block in each block, so that they will sum to the block index.
  std::unique_ptr<float> blocks(new float[vcsmc::kFrameSizeBytes]);
  int block_counter = 0;
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 10; ++j) {
      int pixel_counter = -511;
      float* block_ptr = blocks.get() +
          (((i * vcsmc::kTargetFrameWidthPixels) + j) * 32);
      for (size_t y = 0; y < 32; ++y) {
        for (size_t x = 0; x < 32; ++x) {
          if (pixel_counter == 512) {
            *block_ptr = static_cast<float>(block_counter);
            ++block_counter;
          } else {
            *block_ptr = static_cast<float>(pixel_counter);
          }
          ++pixel_counter;
          ++block_ptr;
        }
        block_ptr += vcsmc::kTargetFrameWidthPixels - 32;
      }
    }
  }

  ASSERT_EQ(cudaSuccess, cudaMemcpy(input_device, blocks.get(),
      vcsmc::kFrameSizeBytes * sizeof(float), cudaMemcpyHostToDevice));
  vcsmc::ComputeBlockSum<<<grid_dim, block_dim, 1024 * sizeof(float)>>>(
      input_device, results_device);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(results.get(), results_device,
      60 * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < 60; ++i) {
    EXPECT_EQ(static_cast<float>(i), results.get()[i]);
  }

  ASSERT_EQ(cudaSuccess, cudaFree(results_device));
  ASSERT_EQ(cudaSuccess, cudaFree(input_device));
}

}  // namespace vcsmc
