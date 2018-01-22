#include "ciede_2k.h"

#include <gtest/gtest.h>

namespace vcsmc {

// Test example data taken from the helpful implementation notes and advice in:
// http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
// "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary
// Test Data, and Mathematical Observations", G. Sharma, W. Wu, E. N. Dalal,
// Color Research and Application, vol. 30. No. 1, February 2005
TEST(ColorTest, Ciede2kSharmaWuBalalTestData) {
  const size_t kSWBTestDataCount = 34;

  const float laba_a[kSWBTestDataCount * 4] = {
    50.0000,   2.6772, -79.7751,   1.0000,
    50.0000,   3.1571, -77.2803,   1.0000,
    50.0000,   2.8361, -74.0200,   1.0000,
    50.0000,  -1.3802, -84.2814,   1.0000,
    50.0000,  -1.1848, -84.8006,   1.0000,
    50.0000,  -0.9009, -85.5211,   1.0000,
    50.0000,   0.0000,   0.0000,   1.0000,
    50.0000,  -1.0000,   2.0000,   1.0000,
    50.0000,   2.4900,  -0.0010,   1.0000,
    50.0000,   2.4900,  -0.0010,   1.0000,
    50.0000,   2.4900,  -0.0010,   1.0000,
    50.0000,   2.4900,  -0.0010,   1.0000,
    50.0000,  -0.0010,   2.4900,   1.0000,
    50.0000,  -0.0010,   2.4900,   1.0000,
    50.0000,  -0.0010,   2.4900,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    50.0000,   2.5000,   0.0000,   1.0000,
    60.2574, -34.0099,  36.2677,   1.0000,
    63.0109, -31.0961,  -5.8663,   1.0000,
    61.2901,   3.7196,  -5.3901,   1.0000,
    35.0831, -44.1164,   3.7933,   1.0000,
    22.7233,  20.0904, -46.6940,   1.0000,
    36.4612,  47.8580,  18.3852,   1.0000,
    90.8027,  -2.0831,   1.4410,   1.0000,
    90.9257,  -0.5406,  -0.9208,   1.0000,
     6.7747,  -0.2908,  -2.4247,   1.0000,
     2.0776,   0.0795,  -1.1350,   1.0000
  };

  const float laba_b[kSWBTestDataCount * 4] = {
    50.0000,   0.0000, -82.7485,   1.0000,
    50.0000,   0.0000, -82.7485,   1.0000,
    50.0000,   0.0000, -82.7485,   1.0000,
    50.0000,   0.0000, -82.7485,   1.0000,
    50.0000,   0.0000, -82.7485,   1.0000,
    50.0000,   0.0000, -82.7485,   1.0000,
    50.0000,  -1.0000,   2.0000,   1.0000,
    50.0000,   0.0000,   0.0000,   1.0000,
    50.0000,  -2.4900,   0.0009,   1.0000,
    50.0000,  -2.4900,   0.0010,   1.0000,
    50.0000,  -2.4900,   0.0011,   1.0000,
    50.0000,  -2.4900,   0.0012,   1.0000,
    50.0000,   0.0009,  -2.4900,   1.0000,
    50.0000,   0.0010,  -2.4900,   1.0000,
    50.0000,   0.0011,  -2.4900,   1.0000,
    50.0000,   0.0000,  -2.5000,   1.0000,
    73.0000,  25.0000, -18.0000,   1.0000,
    61.0000,  -5.0000,  29.0000,   1.0000,
    56.0000, -27.0000,  -3.0000,   1.0000,
    58.0000,  24.0000,  15.0000,   1.0000,
    50.0000,   3.1736,   0.5854,   1.0000,
    50.0000,   3.2972,   0.0000,   1.0000,
    50.0000,   1.8634,   0.5757,   1.0000,
    50.0000,   3.2592,   0.3350,   1.0000,
    60.4626, -34.1751,  39.4387,   1.0000,
    62.8187, -29.7946,  -4.0864,   1.0000,
    61.4292,   2.2480,  -4.9620,   1.0000,
    35.0232, -40.0716,   1.5901,   1.0000,
    23.0331,  14.9730, -42.5619,   1.0000,
    36.2715,  50.5065,  21.2231,   1.0000,
    91.1528,  -1.6435,   0.0447,   1.0000,
    88.6381,  -0.8985,  -0.7239,   1.0000,
     5.8714,  -0.0985,  -2.2286,   1.0000,
     0.9033,  -0.0636,  -0.5514,   1.0000
  };

  const float expected_results[kSWBTestDataCount] = {
     2.0425,   2.8615,   3.4412,   1.0000,   1.0000,   1.0000,
     2.3669,   2.3669,   7.1792,   7.1792,   7.2195,   7.2195,
     4.8045,   4.8045,   4.7461,   4.3065,  27.1492,  22.8977,
    31.9030,  19.4535,   1.0000,   1.0000,   1.0000,   1.0000,
     1.2644,   1.2630,   1.8731,   1.8645,   2.0373,   1.4146,
     1.4441,   1.5381,   0.6377,   0.9082
  };

  float4* laba_a_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&laba_a_device,
      sizeof(float) * 4 * kSWBTestDataCount));
  float4* laba_b_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&laba_b_device,
      sizeof(float) * 4 * kSWBTestDataCount));
  float* results_device;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&results_device,
      sizeof(float) * kSWBTestDataCount));

  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_a_device, laba_a,
      sizeof(float) * 4 * kSWBTestDataCount, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(laba_b_device, laba_b,
      sizeof(float) * 4 * kSWBTestDataCount, cudaMemcpyHostToDevice));

  dim3 dim_grid(1);
  dim3 dim_block(kSWBTestDataCount);
  vcsmc::Ciede2k<<<dim_grid, dim_block>>>(laba_a_device, laba_b_device,
      results_device);

  std::unique_ptr<float> results(new float[kSWBTestDataCount]);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(results.get(), results_device,
      sizeof(float) * kSWBTestDataCount, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kSWBTestDataCount; ++i) {
    EXPECT_NEAR(expected_results[i], results.get()[i], 0.01f);
  }

  ASSERT_EQ(cudaSuccess, cudaFree(results_device));
  ASSERT_EQ(cudaSuccess, cudaFree(laba_b_device));
  ASSERT_EQ(cudaSuccess, cudaFree(laba_a_device));
}

}
