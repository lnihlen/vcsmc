#include "mssim.h"

#include <cstring>
#include <cmath>

#include "gtest/gtest.h"

#include "color.h"

namespace {

std::unique_ptr<float> MakeTestInput(float slope = 1.0f,
                                     float intercept = 0.0f) {
  std::unique_ptr<float> nl_input(new float[vcsmc::kLBufferSize]);
  float* nl_ptr = nl_input.get();

  // Populate each block of pixels starting from upper left hand corner with
  // (i / kWindowSize, i / kWindowSize + 1000, -(i / kWindowSize + 1000))
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      *nl_ptr = (static_cast<float>(
          ((i / vcsmc::kWindowSize) *
          (vcsmc::kTargetFrameWidthPixels / vcsmc::kWindowSize)) +
          (j / vcsmc::kWindowSize)) * slope) + intercept;
      ++nl_ptr;
    }
  }

  return nl_input;
}

// Assumes MakeTestInput was called with default arguments.
std::unique_ptr<float> MakeTestMean() {
  std::unique_ptr<float> mean(new float[vcsmc::kLBufferSize]);
  float* mean_ptr = mean.get();

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
      *mean_ptr = static_cast<float>(
          x + (y * vcsmc::kTargetFrameWidthPixels / vcsmc::kWindowSize))
          / static_cast<float>(vcsmc::kWindowSize);
      ++mean_ptr;
    }
  }

  return mean;
}

std::unique_ptr<float> ComputeTestMean(const float* input) {
  std::unique_ptr<float> mean(new float[vcsmc::kLBufferSize]);
  float* mean_ptr = mean.get();

  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      uint32 mean_offset = (i * vcsmc::kTargetFrameWidthPixels) + j;
      float sum = 0.0f;
      float n = 0.0f;
      for (size_t y = 0; y < vcsmc::kWindowSize; ++y) {
        if (i + y >= vcsmc::kFrameHeightPixels) continue;
        uint32 offset = mean_offset + (y * vcsmc::kTargetFrameWidthPixels);
        for (size_t x = 0; x < vcsmc::kWindowSize; ++x) {
          if (j + x >= vcsmc::kTargetFrameWidthPixels) continue;
          sum += input[offset];
          n += 1.0;
          ++offset;
        }
      }
      mean_ptr[mean_offset] = sum / n;
    }
  }

  return mean;
}

std::unique_ptr<float> ComputeTestVariance(const float* input,
                                           const float* input_mean) {
  std::unique_ptr<float> variance(new float[vcsmc::kLBufferSize]);
  float* variance_ptr = variance.get();
  std::memset(variance_ptr, 0, vcsmc::kLBufferSizeBytes);

  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      uint32 variance_offset = (i * vcsmc::kTargetFrameWidthPixels) + j;
      float sum = 0.0f;
      float mean = input_mean[variance_offset];
      float n = 0.0f;
      for (size_t y = 0; y < vcsmc::kWindowSize; ++y) {
        if (i + y >= vcsmc::kFrameHeightPixels) continue;
        uint32 offset = variance_offset + (y * vcsmc::kTargetFrameWidthPixels);
        for (size_t x = 0; x < vcsmc::kWindowSize; ++x) {
          if (j + x >= vcsmc::kTargetFrameWidthPixels) continue;
          float del = input[offset] - mean;
          sum += del * del;
          n += 1.0;
          ++offset;
        }
      }
      float n_minus_one = std::max(1.0f, n - 1.0f);
      variance_ptr[variance_offset] = sum / n_minus_one;
    }
  }

  return variance;
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

std::unique_ptr<float> ComputeTestSSIM(const float* mean_a,
                                       const float* variance_a,
                                       const float* mean_b,
                                       const float* variance_b,
                                       const float* covariance_ab) {
  std::unique_ptr<float> ssim(new float[vcsmc::kLBufferSize]);
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      size_t index = (i * vcsmc::kTargetFrameWidthPixels) + j;
      ssim.get()[index] = Ssim(mean_a[index],
                               variance_a[index],
                               mean_b[index],
                               variance_b[index],
                               covariance_ab[index]);
    }
  }

  return ssim;
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
  std::unique_ptr<float> nl_mean_generated = MakeTestMean();
  std::unique_ptr<float> nl_input = MakeTestInput();
  std::unique_ptr<float> nl_mean_computed = ComputeTestMean(nl_input.get());

  ASSERT_EQ(0, std::memcmp(nl_mean_generated.get(),
                           nl_mean_computed.get(),
                           vcsmc::kLBufferSizeBytes));
}

TEST(MssimTest, LabaToNormalizedLTest) {
  std::unique_ptr<float> laba_input(new float[vcsmc::kLabaBufferSize]);
  for (size_t i = 0; i < vcsmc::kLabaBufferSize; i += 4) {
    float i_float = static_cast<float>(i / 4);
    laba_input.get()[i] = i_float;
    laba_input.get()[i + 1] = -i_float + 3.0;
    laba_input.get()[i + 2] = 61440.0f - i_float;
    laba_input.get()[i + 3] = 1.0f;
  }

  float4* laba_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&laba_device, vcsmc::kLabaBufferSizeBytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_device, laba_input.get(),
      vcsmc::kLabaBufferSizeBytes, cudaMemcpyHostToDevice));
  float* nl_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nl_device, vcsmc::kLabaBufferSizeBytes));

  dim3 dim_grid(vcsmc::kTargetFrameWidthPixels / 32,
                vcsmc::kFrameHeightPixels / 32);
  dim3 dim_block(32, 32);
  vcsmc::LabaToNormalizedL<<<dim_grid, dim_block>>>(laba_device, nl_device);

  std::unique_ptr<float> nl_input(new float[vcsmc::kLBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nl_input.get(), nl_device,
      vcsmc::kLBufferSizeBytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < vcsmc::kLBufferSize; ++i) {
    EXPECT_EQ(static_cast<float>(i) / vcsmc::kMaxLab,
              nl_input.get()[i]);
  }

  ASSERT_EQ(cudaSuccess, cudaFree(nl_device));
  ASSERT_EQ(cudaSuccess, cudaFree(laba_device));
}

TEST(MssimTest, MeanTest) {
  std::unique_ptr<float> nl_input = MakeTestInput();

  // Copy laba buffer to device, compute mean, copy it back.
  float* nl_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nl_device, vcsmc::kLBufferSizeBytes));
  float* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLBufferSizeBytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nl_device, nl_input.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));

  dim3 dim_grid(vcsmc::kTargetFrameWidthPixels / 32,
                vcsmc::kFrameHeightPixels / 32);
  dim3 dim_block(32, 32);
  vcsmc::ComputeMean<<<dim_grid, dim_block>>>(nl_device, mean_device);

  std::unique_ptr<float> mean_output(new float[vcsmc::kLBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_output.get(), mean_device,
      vcsmc::kLBufferSizeBytes, cudaMemcpyDeviceToHost));

  std::unique_ptr<float> mean_generated = MakeTestMean();

  EXPECT_EQ(0, std::memcmp(mean_output.get(),
                           mean_generated.get(),
                           vcsmc::kLBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaFree(nl_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
}

TEST(MssimTest, VarianceTest) {
  std::unique_ptr<float> nl_input = MakeTestInput();
  std::unique_ptr<float> nl_mean = MakeTestMean();

  float* nl_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nl_device, vcsmc::kLBufferSizeBytes));
  float* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLBufferSizeBytes));
  float* variance_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&variance_device,
        vcsmc::kLBufferSizeBytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nl_device, nl_input.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_device, nl_mean.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));

  dim3 dim_grid(vcsmc::kTargetFrameWidthPixels / 32,
                vcsmc::kFrameHeightPixels / 32);
  dim3 dim_block(32, 32);
  vcsmc::ComputeVariance<<<dim_grid, dim_block>>>(
      nl_device, mean_device, variance_device);

  std::unique_ptr<float> variance_output(new float[vcsmc::kLBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(variance_output.get(), variance_device,
      vcsmc::kLBufferSizeBytes, cudaMemcpyDeviceToHost));

  std::unique_ptr<float> variance_computed =
      ComputeTestVariance(nl_input.get(), nl_mean.get());

  EXPECT_EQ(0, std::memcmp(variance_computed.get(),
                           variance_output.get(),
                           vcsmc::kLBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaFree(nl_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
  ASSERT_EQ(cudaSuccess, cudaFree(variance_device));
}

TEST(MssimTest, CovarianceTest) {
  std::unique_ptr<float> nl_input = MakeTestInput();
  std::unique_ptr<float> nl_mean = MakeTestMean();

  float* nl_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nl_device, vcsmc::kLBufferSizeBytes));
  float* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLBufferSizeBytes));
  float* covariance_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&covariance_device,
                                    vcsmc::kLBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(nl_device, nl_input.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_device, nl_mean.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));

  // Covariance of variable with itself should be equal to variance, or
  // standard deviation squared.
  dim3 dim_grid(vcsmc::kTargetFrameWidthPixels / 32,
                vcsmc::kFrameHeightPixels / 32);
  dim3 dim_block(32, 32);
  vcsmc::ComputeCovariance<<<dim_grid, dim_block>>>(
      nl_device, mean_device, nl_device, mean_device, covariance_device);

  std::unique_ptr<float> covariance(new float[vcsmc::kLBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kLBufferSizeBytes, cudaMemcpyDeviceToHost));

  std::unique_ptr<float> variance_computed =
      ComputeTestVariance(nl_input.get(), nl_mean.get());

  EXPECT_EQ(0, std::memcmp(variance_computed.get(),
                           covariance.get(),
                           vcsmc::kLBufferSizeBytes));

  // cov(X + a, Y + b) = cov(X, Y) for constants a,b, random variables X, Y.
  std::unique_ptr<float> nl_plus_a = MakeTestInput(1.0f, 2.0f);
  std::unique_ptr<float> nl_mean_a = ComputeTestMean(nl_plus_a.get());
  std::unique_ptr<float> nl_plus_b = MakeTestInput(1.0f, -4.0f);
  std::unique_ptr<float> nl_mean_b = ComputeTestMean(nl_plus_b.get());

  float* nl_a_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nl_a_device,
                                    vcsmc::kLBufferSizeBytes));
  float* mean_a_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_a_device,
                                    vcsmc::kLBufferSizeBytes));
  float* nl_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&nl_b_device,
                                    vcsmc::kLBufferSizeBytes));
  float* mean_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_b_device,
                                    vcsmc::kLBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(nl_a_device, nl_plus_a.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_a_device, nl_mean_a.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nl_b_device, nl_plus_b.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_b_device, nl_mean_b.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));

  vcsmc::ComputeCovariance<<<dim_grid, dim_block>>>(
      nl_a_device, mean_a_device, nl_b_device, mean_b_device,
      covariance_device);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kLBufferSizeBytes, cudaMemcpyDeviceToHost));

  EXPECT_EQ(0, std::memcmp(variance_computed.get(),
                           covariance.get(),
                           vcsmc::kLBufferSizeBytes));

  // cov(aX, bY) = ab cov(X,Y) for constants a,b, random variables X, Y.
  std::unique_ptr<float> nl_times_a = MakeTestInput(-5.0f, 0.0f);
  nl_mean_a = ComputeTestMean(nl_times_a.get());
  std::unique_ptr<float> nl_times_b = MakeTestInput(3.0f, 0.0f);
  nl_mean_b = ComputeTestMean(nl_times_b.get());

  ASSERT_EQ(cudaSuccess, cudaMemcpy(nl_a_device, nl_times_a.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_a_device, nl_mean_a.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(nl_b_device, nl_times_b.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_b_device, nl_mean_b.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));

  vcsmc::ComputeCovariance<<<dim_grid, dim_block>>>(
      nl_a_device, mean_a_device, nl_b_device, mean_b_device,
      covariance_device);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance.get(), covariance_device,
      vcsmc::kLBufferSizeBytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < vcsmc::kLBufferSize; ++i) {
    EXPECT_GT(16, ulp_delta(-15.0f * variance_computed.get()[i],
                            covariance.get()[i]));
  }

  ASSERT_EQ(cudaSuccess, cudaFree(nl_a_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_a_device));
  ASSERT_EQ(cudaSuccess, cudaFree(nl_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(covariance_device));
  ASSERT_EQ(cudaSuccess, cudaFree(mean_device));
  ASSERT_EQ(cudaSuccess, cudaFree(nl_device));
}

TEST(MssimTest, SsimTest) {
  // Check that SSIM(x, x) = 1.0.
  std::unique_ptr<float> nl_input = MakeTestInput();
  std::unique_ptr<float> nl_mean = MakeTestMean();
  std::unique_ptr<float> nl_variance =
      ComputeTestVariance(nl_input.get(), nl_mean.get());

  float* mean_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_device, vcsmc::kLBufferSizeBytes));
  float* variance_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&variance_device,
                                    vcsmc::kLBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_device, nl_mean.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(variance_device, nl_variance.get(),
        vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));

  float* ssim_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&ssim_device, vcsmc::kLBufferSizeBytes));

  dim3 dim_grid(vcsmc::kTargetFrameWidthPixels / 32,
                vcsmc::kFrameHeightPixels / 32);
  dim3 dim_block(32, 32);
  vcsmc::ComputeSSIM<<<dim_grid, dim_block>>>(
      mean_device,
      variance_device,
      mean_device,
      variance_device,
      variance_device,
      ssim_device);

  std::unique_ptr<float> ssim(new float[vcsmc::kLBufferSize]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(ssim.get(), ssim_device,
        vcsmc::kLBufferSizeBytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < vcsmc::kLBufferSize; ++i) {
    EXPECT_GT(4, ulp_delta(1.0f, ssim.get()[i]));
  }

  // Compare device-computed SSIM with host-computed values.
  std::unique_ptr<float> nl_b = MakeTestInput(-11.0f, 4.0f);
  std::unique_ptr<float> nl_b_mean = ComputeTestMean(nl_b.get());
  std::unique_ptr<float> nl_b_variance =
      ComputeTestVariance(nl_b.get(), nl_b_mean.get());

  // Re-use cov(aX, X) = a variance(X) to build covariance values.
  std::unique_ptr<float> covariance(new float[vcsmc::kLBufferSize]);
  for (size_t i = 0; i < vcsmc::kLBufferSize; ++i) {
    covariance.get()[i] = nl_variance.get()[i] * -11.0f;
  }

  float* mean_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&mean_b_device,
                                    vcsmc::kLabaBufferSizeBytes));
  float* variance_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&variance_b_device,
                                    vcsmc::kLabaBufferSizeBytes));
  float* covariance_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&covariance_device,
                                    vcsmc::kLabaBufferSizeBytes));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(mean_b_device, nl_b_mean.get(),
      vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(variance_b_device, nl_b_variance.get(),
      vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(covariance_device, covariance.get(),
      vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));

  vcsmc::ComputeSSIM<<<dim_grid, dim_block>>>(
      mean_device,
      variance_device,
      mean_b_device,
      variance_b_device,
      covariance_device,
      ssim_device);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(ssim.get(), ssim_device,
      vcsmc::kLBufferSizeBytes, cudaMemcpyDeviceToHost));

  std::unique_ptr<float> ssim_computed = ComputeTestSSIM(
      nl_mean.get(),
      nl_variance.get(),
      nl_b_mean.get(),
      nl_b_variance.get(),
      covariance.get());

  for (size_t i = 0; i < vcsmc::kLBufferSize; ++i) {
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
  std::unique_ptr<float> zeros(new float[vcsmc::kLBufferSize]);
  std::memset(zeros.get(), 0, vcsmc::kLBufferSizeBytes);

  float* input_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&input_device, vcsmc::kLBufferSizeBytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(input_device, zeros.get(),
      vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));

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
  std::unique_ptr<float> blocks(new float[vcsmc::kLBufferSize]);
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
      vcsmc::kLBufferSizeBytes, cudaMemcpyHostToDevice));
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
