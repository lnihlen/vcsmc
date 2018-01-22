#include "mssim.h"

#include <cstring>
#include <cmath>

#include "gtest/gtest.h"

namespace {

std::unique_ptr<float> MakePaddedTestInput(float slope = 1.0f,
                                           float intercept = 0.0f) {
  std::unique_ptr<float> laba_input(new float[vcsmc::kLabaBufferSize]);
  float* laba_ptr = laba_input.get();

  // Zero laba buffer to include padding.
  std::memset(laba_ptr, 0, vcsmc::kLabaBufferSizeBytes);

  // Populate each block of pixels starting from upper left hand corner with
  // (i / kWindowSize, i / kWindowSize + 1000, -(i / kWindowSize + 1000))
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      float val = (static_cast<float>(
          ((i / vcsmc::kWindowSize) *
          (vcsmc::kTargetFrameWidthPixels / vcsmc::kWindowSize)) +
          (j / vcsmc::kWindowSize)) * slope) + intercept;
      *laba_ptr = val;
      ++laba_ptr;
      *laba_ptr = val + 0.5f;
      ++laba_ptr;
      *laba_ptr = -(val + 0.5f);
      ++laba_ptr;
      *laba_ptr = 1.0f;
      ++laba_ptr;
    }
    // Skip padding on right.
    laba_ptr +=  4 * vcsmc::kWindowSize;
  }

  return laba_input;
}

// Assumes MakePaddedTestInput was called with default arguments.
std::unique_ptr<float> MakePaddedTestMean() {
  std::unique_ptr<float> mean(new float[vcsmc::kLabaBufferSize]);
  float* mean_ptr = mean.get();
  std::memset(mean_ptr, 0, vcsmc::kLabaBufferSizeBytes);

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
  std::unique_ptr<float> mean(new float[vcsmc::kLabaBufferSize]);
  float* mean_ptr = mean.get();
  std::memset(mean_ptr, 0, vcsmc::kLabaBufferSizeBytes);

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
  std::unique_ptr<float> stddevsq(new float[vcsmc::kLabaBufferSize]);
  float* stddevsq_ptr = stddevsq.get();
  std::memset(stddevsq_ptr, 0, vcsmc::kLabaBufferSizeBytes);

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

float Ssim(float mean_a,
           float variance_a,
           float mean_b,
           float variance_b,
           float covariance_ab) {
  const float kC1 = 0.0001f;
  const float kC2 = 0.0009f;
  return std::fabs((((2.0f * mean_a * mean_b) + kC1) *
                   ((2.0f * covariance_ab) + kC2)) /
                  (((mean_a * mean_a) + (mean_b * mean_b) + kC1) *
                   (variance_a + variance_b + kC2)));
}

std::unique_ptr<float> ComputeTestSSIM(const float* padded_mean_a,
                                       const float* padded_variance_a,
                                       const float* padded_mean_b,
                                       const float* padded_variance_b,
                                       const float* padded_covariance_ab) {
  std::unique_ptr<float> ssim(new float[vcsmc::kFrameSizeBytes]);
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      size_t padded_index =
        ((i * (vcsmc::kTargetFrameWidthPixels + vcsmc::kWindowSize)) + j) * 4;
      float ssim_y = Ssim(padded_mean_a[padded_index],
                          padded_variance_a[padded_index],
                          padded_mean_b[padded_index],
                          padded_variance_b[padded_index],
                          padded_covariance_ab[padded_index]);

      float ssim_u = Ssim(padded_mean_a[padded_index + 1],
                          padded_variance_a[padded_index + 1],
                          padded_mean_b[padded_index + 1],
                          padded_variance_b[padded_index + 1],
                          padded_covariance_ab[padded_index + 1]);

      float ssim_v = Ssim(padded_mean_a[padded_index + 2],
                          padded_variance_a[padded_index + 2],
                          padded_mean_b[padded_index + 2],
                          padded_variance_b[padded_index + 2],
                          padded_covariance_ab[padded_index + 2]);

      ssim.get()[(i * vcsmc::kTargetFrameWidthPixels) + j] =
          (0.5f * ssim_y) +
          (0.25f * ssim_u) +
          (0.25f * ssim_v);
    }
  }

  return ssim;
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
  ASSERT_EQ(vcsmc::kLabaBufferSize, offset);
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
  std::unique_ptr<float> laba_input = MakePaddedTestInput();
  ValidatePaddingWeights(laba_input.get());

  std::unique_ptr<float> laba_mean_generated = MakePaddedTestMean();
  ValidatePaddingWeights(laba_mean_generated.get());

  std::unique_ptr<float> laba_mean_computed =
    ComputePaddedMean(laba_input.get());
  ValidatePaddingWeights(laba_mean_computed.get());

  ASSERT_EQ(0, std::memcmp(laba_mean_generated.get(),
                           laba_mean_computed.get(),
                           vcsmc::kLabaBufferSizeBytes));

  std::unique_ptr<float> laba_stddevsq_computed =
      ComputePaddedStandardDeviationSquared(laba_input.get(),
                                            laba_mean_generated.get());
  ValidatePaddingWeights(laba_stddevsq_computed.get());
}


TEST(MssimTest, MeanTest) {
  std::unique_ptr<float> laba_input = MakePaddedTestInput();

  // Copy laba buffer to device, compute mean, copy it back.
  float4* laba_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&laba_device, vcsmc::kLabaBufferSizeBytes));
  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLabaBufferSizeBytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_device, laba_input.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));

  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  dim3 image_dim_block(16, 16);
  vcsmc::ComputeLocalMean<<<image_dim_grid, image_dim_block>>>(
      laba_device, mean_device);

  std::unique_ptr<float> mean_output(new float[vcsmc::kLabaBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_output.get(), mean_device,
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyDeviceToHost));

  std::unique_ptr<float> laba_mean_generated = MakePaddedTestMean();

  EXPECT_EQ(0, std::memcmp(mean_output.get(),
                           laba_mean_generated.get(),
                           vcsmc::kLabaBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaFree(laba_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
}

TEST(MssimTest, StandardDeviationSquaredTest) {
  std::unique_ptr<float> laba_input = MakePaddedTestInput();
  std::unique_ptr<float> laba_mean = MakePaddedTestMean();

  float4* laba_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&laba_device, vcsmc::kLabaBufferSizeBytes));
  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLabaBufferSizeBytes));
  float4* stddevsq_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&stddevsq_device,
      vcsmc::kLabaBufferSizeBytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_device, laba_input.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_device, laba_mean.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));

  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  dim3 image_dim_block(16, 16);
  vcsmc::ComputeLocalStdDevSquared<<<image_dim_grid, image_dim_block>>>(
      laba_device, mean_device, stddevsq_device);

  std::unique_ptr<float> stddevsq_output(new float[vcsmc::kLabaBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(stddevsq_output.get(), stddevsq_device,
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(stddevsq_output.get());

  std::unique_ptr<float> laba_stddevsq_computed =
      ComputePaddedStandardDeviationSquared(laba_input.get(), laba_mean.get());

  EXPECT_EQ(0, std::memcmp(laba_stddevsq_computed.get(),
                           stddevsq_output.get(),
                           vcsmc::kLabaBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaFree(laba_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
  ASSERT_EQ(cudaSuccess, cudaFree(stddevsq_device));
}

TEST(MssimTest, CovarianceTest) {
  std::unique_ptr<float> laba_input = MakePaddedTestInput();
  std::unique_ptr<float> laba_mean = MakePaddedTestMean();

  float4* laba_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&laba_device, vcsmc::kLabaBufferSizeBytes));
  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLabaBufferSizeBytes));
  float4* covariance_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&covariance_device,
                                    vcsmc::kLabaBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_device, laba_input.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_device, laba_mean.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));

  // Covariance of variable with itself should be equal to variance, or
  // standard deviation squared.
  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  dim3 image_dim_block(16, 16);
  vcsmc::ComputeLocalCovariance<<<image_dim_grid, image_dim_block>>>(
      laba_device, mean_device, laba_device, mean_device, covariance_device);

  std::unique_ptr<float> covariance(new float[vcsmc::kLabaBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(covariance.get());

  std::unique_ptr<float> variance_computed =
      ComputePaddedStandardDeviationSquared(laba_input.get(), laba_mean.get());

  EXPECT_EQ(0, std::memcmp(variance_computed.get(),
                           covariance.get(),
                           vcsmc::kLabaBufferSizeBytes));

  // cov(X + a, Y + b) = cov(X, Y) for constants a,b, random variables X, Y.
  std::unique_ptr<float> laba_plus_a = MakePaddedTestInput(1.0f, 2.0f);
  std::unique_ptr<float> laba_mean_a = ComputePaddedMean(laba_plus_a.get());
  std::unique_ptr<float> laba_plus_b = MakePaddedTestInput(1.0f, -4.0f);
  std::unique_ptr<float> laba_mean_b = ComputePaddedMean(laba_plus_b.get());

  float4* laba_a_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&laba_a_device,
                                    vcsmc::kLabaBufferSizeBytes));
  float4* mean_a_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_a_device,
                                    vcsmc::kLabaBufferSizeBytes));
  float4* laba_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&laba_b_device,
                                    vcsmc::kLabaBufferSizeBytes));
  float4* mean_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_b_device,
                                    vcsmc::kLabaBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_a_device, laba_plus_a.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_a_device, laba_mean_a.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_b_device, laba_plus_b.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_b_device, laba_mean_b.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));

  vcsmc::ComputeLocalCovariance<<<image_dim_grid, image_dim_block>>>(
      laba_a_device, mean_a_device, laba_b_device, mean_b_device,
      covariance_device);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(covariance.get());

  EXPECT_EQ(0, std::memcmp(variance_computed.get(),
                           covariance.get(),
                           vcsmc::kLabaBufferSizeBytes));

  // cov(aX, bY) = ab cov(X,Y) for constants a,b, random variables X, Y.
  std::unique_ptr<float> laba_times_a = MakePaddedTestInput(-5.0f, 0.0f);
  laba_mean_a = ComputePaddedMean(laba_times_a.get());
  std::unique_ptr<float> laba_times_b = MakePaddedTestInput(3.0f, 0.0f);
  laba_mean_b = ComputePaddedMean(laba_times_b.get());

  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_a_device, laba_times_a.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_a_device, laba_mean_a.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_b_device, laba_times_b.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_b_device, laba_mean_b.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));

  vcsmc::ComputeLocalCovariance<<<image_dim_grid, image_dim_block>>>(
      laba_a_device, mean_a_device, laba_b_device, mean_b_device,
      covariance_device);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyDeviceToHost));
  ValidatePaddingWeights(covariance.get());

  for (size_t i = 0; i < vcsmc::kLabaBufferSize; i += 4) {
    EXPECT_GT(16, ulp_delta(-15.0f * variance_computed.get()[i],
                           covariance.get()[i]));
    EXPECT_GT(16, ulp_delta(-15.0f * variance_computed.get()[i + 1],
                           covariance.get()[i + 1]));
    EXPECT_GT(16, ulp_delta(-15.0f * variance_computed.get()[i + 2],
                           covariance.get()[i + 2]));
    EXPECT_EQ(variance_computed.get()[i + 3],
              covariance.get()[i + 3]);
  }

  ASSERT_EQ(cudaSuccess, cudaFree(laba_a_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_a_device));
  ASSERT_EQ(cudaSuccess, cudaFree(laba_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(covariance_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
  ASSERT_EQ(cudaSuccess, cudaFree(laba_device));
}

TEST(MssimTest, SsimTest) {
  // Check that SSIM(x, x) = 1.0.
  std::unique_ptr<float> laba_input = MakePaddedTestInput();
  std::unique_ptr<float> laba_mean = MakePaddedTestMean();
  std::unique_ptr<float> laba_variance =
      ComputePaddedStandardDeviationSquared(laba_input.get(), laba_mean.get());

  float4* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLabaBufferSizeBytes));
  float4* variance_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&variance_device,
                                    vcsmc::kLabaBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_device, laba_mean.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(variance_device, laba_variance.get(),
        vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));

  float* ssim_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&ssim_device,
        vcsmc::kFrameSizeBytes * sizeof(float)));

  dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                      (vcsmc::kFrameHeightPixels / 16) + 1);
  dim3 image_dim_block(16, 16);
  vcsmc::ComputeSSIM<<<image_dim_grid, image_dim_block>>>(
      mean_device,
      variance_device,
      mean_device,
      variance_device,
      variance_device,
      ssim_device);

  std::unique_ptr<float> ssim(new float[vcsmc::kFrameSizeBytes]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(ssim.get(), ssim_device,
        vcsmc::kFrameSizeBytes * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < vcsmc::kFrameSizeBytes; ++i) {
    EXPECT_EQ(1.0f, ssim.get()[i]);
  }

  // Compare device-computed SSIM with host-computed values.
  std::unique_ptr<float> laba_b = MakePaddedTestInput(-11.0f, 4.0f);
  std::unique_ptr<float> laba_b_mean = ComputePaddedMean(laba_b.get());
  std::unique_ptr<float> laba_b_variance =
      ComputePaddedStandardDeviationSquared(laba_b.get(), laba_b_mean.get());

  // Re-use cov(aX, X) = a variance(X) to build covariance values.
  std::unique_ptr<float> covariance(new float[vcsmc::kLabaBufferSize]);
  for (size_t i = 0; i < vcsmc::kLabaBufferSize; i += 4) {
    covariance.get()[i] = laba_variance.get()[i] * -11.0f;
    covariance.get()[i + 1] = laba_variance.get()[i] * -11.0f;
    covariance.get()[i + 2] = laba_variance.get()[i] * -11.0f;
    covariance.get()[i + 3] = laba_variance.get()[i];
  }

  float4* mean_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_b_device,
                                    vcsmc::kLabaBufferSizeBytes));
  float4* variance_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&variance_b_device,
                                    vcsmc::kLabaBufferSizeBytes));
  float4* covariance_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&covariance_device,
                                    vcsmc::kLabaBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_b_device, laba_b_mean.get(),
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(variance_b_device, laba_b_variance.get(),
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance_device, covariance.get(),
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));

  vcsmc::ComputeSSIM<<<image_dim_grid, image_dim_block>>>(
      mean_device,
      variance_device,
      mean_b_device,
      variance_b_device,
      covariance_device,
      ssim_device);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(ssim.get(), ssim_device,
      vcsmc::kFrameSizeBytes * sizeof(float), cudaMemcpyDeviceToHost));

  std::unique_ptr<float> ssim_computed = ComputeTestSSIM(
      laba_mean.get(),
      laba_variance.get(),
      laba_b_mean.get(),
      laba_b_variance.get(),
      covariance.get());

  for (size_t i = 0; i < vcsmc::kFrameSizeBytes; ++i) {
    EXPECT_GT(4, ulp_delta(ssim_computed.get()[i], ssim.get()[i]));
  }

  ASSERT_EQ(cudaSuccess, cudaFree(covariance_device));
  ASSERT_EQ(cudaSuccess, cudaFree(variance_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(ssim_device));
  ASSERT_EQ(cudaSuccess, cudaFree(variance_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
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
