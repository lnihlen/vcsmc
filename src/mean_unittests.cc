#include "mean.h"
#include <gtest/gtest.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#include "Halide.h"
#pragma clang diagnostic pop

#include <cstring>

namespace vcsmc {

TEST(MeanTest, WeightedMeanOfConstantShouldBeConstant) {
  Halide::Runtime::Buffer<float, 3> mean_input(15, 15, 3);
  Halide::Runtime::Buffer<float, 2> mean_output(1, 1);

  for (size_t i = 0; i < (15 * 15); ++i) {
    mean_input.begin()[i] = 1.0;
    mean_input.begin()[i + (15 * 15)] = 0.0;
    mean_input.begin()[i + (2 * 15 * 15)] = 0.0;
  }

  *mean_output.begin() = 0.0f;

  mean_output.set_min(7, 7);
  mean(mean_input, mean_output);

  EXPECT_NEAR(1.0f, mean_output.begin()[0], 0.001f);
}

TEST(MeanTest, Extract2DKernelFromMean) {
  // Test area is a 15x15 buffer of all 0s with a 1 in the center. Will need
  // a 7 pixel padding all around 15x15 area, so 7 + 15 + 7 or 29px in both
  // dimensions.
  Halide::Runtime::Buffer<float, 3> mean_input(29, 29, 3);
  Halide::Runtime::Buffer<float, 2> mean_output(15, 15);

  std::memset(mean_input.begin(), 0, sizeof(float) * 29 * 29 * 3);
  mean_input.begin()[(14 * 29) + 14] = 1.0;

  std::memset(mean_output.begin(), 0, sizeof(float) * 15 * 15);

  mean_output.set_min(7, 7);
  mean(mean_input, mean_output);

  const float kExpected2DKernelValues[] = {
      0.000215, 0.000376, 0.000602, 0.000886, 0.001197, 0.001484, 0.001688,
      0.001762, 0.001688, 0.001484, 0.001197, 0.000886, 0.000602, 0.000376,
      0.000215, 0.000376, 0.000656, 0.001053, 0.001549, 0.002092, 0.002593,
      0.002950, 0.003079, 0.002950, 0.002593, 0.002092, 0.001549, 0.001053,
      0.000656, 0.000376, 0.000602, 0.001053, 0.001688, 0.002484, 0.003356,
      0.004159, 0.004731, 0.004939, 0.004731, 0.004159, 0.003356, 0.002484,
      0.001688, 0.001053, 0.000602, 0.000886, 0.001549, 0.002484, 0.003657,
      0.004939, 0.006122, 0.006963, 0.007269, 0.006963, 0.006122, 0.004939,
      0.003657, 0.002484, 0.001549, 0.000886, 0.001197, 0.002092, 0.003356,
      0.004939, 0.006671, 0.008268, 0.009405, 0.009818, 0.009405, 0.008268,
      0.006671, 0.004939, 0.003356, 0.002092, 0.001197, 0.001484, 0.002593,
      0.004159, 0.006122, 0.008268, 0.010249, 0.011658, 0.012169, 0.011658,
      0.010249, 0.008268, 0.006122, 0.004159, 0.002593, 0.001484, 0.001688,
      0.002950, 0.004731, 0.006963, 0.009405, 0.011658, 0.013261, 0.013842,
      0.013261, 0.011658, 0.009405, 0.006963, 0.004731, 0.002950, 0.001688,
      0.001762, 0.003079, 0.004939, 0.007269, 0.009818, 0.012169, 0.013842,
      0.014450, 0.013842, 0.012169, 0.009818, 0.007269, 0.004939, 0.003079,
      0.001762, 0.001688, 0.002950, 0.004731, 0.006963, 0.009405, 0.011658,
      0.013261, 0.013842, 0.013261, 0.011658, 0.009405, 0.006963, 0.004731,
      0.002950, 0.001688, 0.001484, 0.002593, 0.004159, 0.006122, 0.008268,
      0.010249, 0.011658, 0.012169, 0.011658, 0.010249, 0.008268, 0.006122,
      0.004159, 0.002593, 0.001484, 0.001197, 0.002092, 0.003356, 0.004939,
      0.006671, 0.008268, 0.009405, 0.009818, 0.009405, 0.008268, 0.006671,
      0.004939, 0.003356, 0.002092, 0.001197, 0.000886, 0.001549, 0.002484,
      0.003657, 0.004939, 0.006122, 0.006963, 0.007269, 0.006963, 0.006122,
      0.004939, 0.003657, 0.002484, 0.001549, 0.000886, 0.000602, 0.001053,
      0.001688, 0.002484, 0.003356, 0.004159, 0.004731, 0.004939, 0.004731,
      0.004159, 0.003356, 0.002484, 0.001688, 0.001053, 0.000602, 0.000376,
      0.000656, 0.001053, 0.001549, 0.002092, 0.002593, 0.002950, 0.003079,
      0.002950, 0.002593, 0.002092, 0.001549, 0.001053, 0.000656, 0.000376,
      0.000215, 0.000376, 0.000602, 0.000886, 0.001197, 0.001484, 0.001688,
      0.001762, 0.001688, 0.001484, 0.001197, 0.000886, 0.000602, 0.000376,
      0.000215
  };

  for (size_t i = 0; i < 15 * 15; ++i) {
    EXPECT_NEAR(kExpected2DKernelValues[i], mean_output.begin()[i], 0.001f);
  }
}

}  // namespace vcsmc

