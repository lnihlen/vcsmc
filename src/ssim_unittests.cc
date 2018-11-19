#include "mean.h"
#include "variance.h"
#include <gtest/gtest.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#include "Halide.h"
#pragma clang diagnostic pop

#include <cstring>

#include "gaussian_kernel.h"

namespace {

class Clamped {
 public:
  Clamped(float* data, int width, int height)
      : data_(data), width_(width), height_(height) { }

  float at(int x, int y) {
    if (x < 0 || x >= width_) return 0.0f;
    if (y < 0 || y >= height_) return 0.0f;
    return data_[(y * width_) + x];
  }

  float* data_;
  int width_;
  int height_;
};

// Fills the supplied output buffer |out| with supplied |width, height| with
// values of parabolic function x*x + y*y, centered at |center_x, center_y|
void ComputeTestCentroid(float* out, size_t width, size_t height,
                         size_t center_x, size_t center_y) {
  float* output_position = out;
  float center_x_float = static_cast<float>(center_x);
  float center_y_float = static_cast<float>(center_y);
  for (size_t i = 0; i < height; ++i) {
    float i_float = static_cast<float>(i);
    float y_term = (i_float - center_y_float) * (i_float - center_y_float);
    for (size_t j = 0; j < width; ++j) {
      float j_float = static_cast<float>(j);
      *output_position = ((j_float - center_x_float) *
                          (j_float - center_x_float)) + y_term;
      ++output_position;
    }
  }
}


// Given a 2D float field at |in| and the kernel, compute the Weighted Mean at
// position |x, y| and return it.
float ComputeWeightedMean(Clamped in, float* kernel, int x, int y) {
  float mean = 0.0;
  size_t kernel_offset = 0;
  int kx = x - 7;
  int ky = y - 7;
  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 15; ++j) {
      mean += kernel[kernel_offset] * in.at(kx + j, ky + i);
      ++kernel_offset;
    }
  }
  return mean;
}

float ComputeWeightedVariance(Clamped in, Clamped mean, float* kernel,
                              int x, int y) {
  float variance = 0.0;
  size_t kernel_offset = 0;
  int kx = x - 7;
  int ky = y - 7;
  for (size_t i = 0; i < 15; ++i) {
    for (size_t j = 0; j < 15; ++j) {
      variance += kernel[kernel_offset] *
                  (in.at(kx + j, ky + i) - mean.at(kx + j, ky + i)) *
                  (in.at(kx + j, ky + i) - mean.at(kx + j, ky + i));
      ++kernel_offset;
    }
  }
  return variance;
}

}  // namespace

namespace vcsmc {

TEST(CentroidTest, TestComputeTestCentroid) {
  float small_centroid[7 * 7];
  ComputeTestCentroid(small_centroid, 7, 7, 3, 3);
  float small_min = small_centroid[0];
  float small_max = small_centroid[(7 * 3) + 3];
  for (size_t i = 0; i < 7 * 7; ++i) {
    small_min = std::min(small_min, small_centroid[i]);
    small_max = std::max(small_max, small_centroid[i]);
  }
  // We expect value in center to be zero and the minimum value.
  ASSERT_EQ(0.0f, small_centroid[(7 * 3) + 3]);
  ASSERT_EQ(small_min, small_centroid[(7 * 3) + 3]);

  // We expect the corners to be maximum value.
  ASSERT_EQ(small_max, small_centroid[0]);
  ASSERT_EQ(small_max, small_centroid[6]);
  ASSERT_EQ(small_max, small_centroid[7 * 6]);
  ASSERT_EQ(small_max, small_centroid[(7 * 6) + 6]);

  // Lastly we expect x/y symmetry.
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      float upper_left = small_centroid[(i * 7) + j];
      // Compare against upper right.
      ASSERT_EQ(upper_left, small_centroid[(i * 7) + 6 - j]);
      // Compare against lower left.
      ASSERT_EQ(upper_left, small_centroid[((6 - i) * 7) + j]);
      // Compare against lower right.
      ASSERT_EQ(upper_left, small_centroid[((6 - i) * 7) + 6 - j]);
    }
  }

  // Now test the relocation properties of the centroid function by moving
  // the centroid around a larger array and comparing the values around there
  // to the ones in small_centroid.
  float large_centroid[16 * 16];
  for (size_t i = 3; i < 16 - 3; ++i) {
    for (size_t j = 3; j < 16 - 3; ++j) {
      ComputeTestCentroid(large_centroid, 16, 16, j, i);
      float* patch = large_centroid + ((i - 3) * 16) + (j - 3);
      for (size_t k = 0; k < 7; ++k) {
        for (size_t l = 0; l < 7; ++l) {
          ASSERT_EQ(small_centroid[(k * 7) + l], *patch);
          ++patch;
        }
        patch += 16 - 7;
      }
    }
  }
}

TEST(MeanTest, WeightedMeanOfConstantShouldBeConstant) {
  Halide::Runtime::Buffer<float, 3> lab_in(15, 15, 3);
  Halide::Runtime::Buffer<float, 2> mean_out(1, 1);

  for (size_t i = 0; i < (15 * 15); ++i) {
    lab_in.begin()[i] = 1.0;
    lab_in.begin()[i + (15 * 15)] = 0.0;
    lab_in.begin()[i + (2 * 15 * 15)] = 0.0;
  }

  *mean_out.begin() = 0.0f;

  mean_out.set_min(7, 7);
  mean(lab_in, mean_out);

  EXPECT_NEAR(1.0f, mean_out.begin()[0], 0.001f);
}

TEST(MeanTest, Extract2DKernelFromMean) {
  Halide::Runtime::Buffer<float, 3> lab_in(15, 15, 3);
  Halide::Runtime::Buffer<float, 2> mean_out(15, 15);

  std::memset(lab_in.begin(), 0, sizeof(float) * 15 * 15 * 3);
  lab_in.begin()[(7 * 15) + 7] = 1.0;

  std::memset(mean_out.begin(), 0, sizeof(float) * 15 * 15);
  mean(lab_in, mean_out);

  Halide::Runtime::Buffer<float, 2> kernel = MakeGaussianKernel();
  for (size_t i = 0; i < 15 * 15; ++i) {
    EXPECT_NEAR(kernel.begin()[i], mean_out.begin()[i], 0.001f);
  }
}

// Test of the test function ComputeWeightedMean().
TEST(MeanTest, Extract2DKernelFromComputedWeightedMean) {
  float lab_in[15 * 15];
  std::memset(lab_in, 0, sizeof(float) * 15 * 15);
  lab_in[(7 * 15) + 7] = 1.0;

  Halide::Runtime::Buffer<float, 2> kernel = MakeGaussianKernel();
  Clamped in(lab_in, 15, 15);

  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 15; ++j) {
      EXPECT_NEAR(kernel.begin()[(i * 15) + j],
                  ComputeWeightedMean(in, kernel.begin(), j, i), 0.001f);
    }
  }
}

// Compare mean function values against test computed values with synthetic
// data.
TEST(MeanTest, CompareMeanToComputedMean) {
  Halide::Runtime::Buffer<float, 3> lab_in(15, 15, 3);
  ComputeTestCentroid(lab_in.begin(), 15, 15, 7, 7);

  Halide::Runtime::Buffer<float, 2> mean_out(15, 15);
  mean(lab_in, mean_out);

  Halide::Runtime::Buffer<float, 2> kernel = MakeGaussianKernel();
  Clamped in(lab_in.begin(), 15, 15);

  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 15; ++j) {
      EXPECT_NEAR(ComputeWeightedMean(in, kernel.begin(), j, i),
                  mean_out.begin()[(i * 15) + j], 0.01);
    }
  }
}

TEST(VarianceTest, CompareVarianceToComputedVariance) {
  Halide::Runtime::Buffer<float, 3> lab_in(15, 15, 3);
  ComputeTestCentroid(lab_in.begin(), 15, 15, 9, 5);

  Halide::Runtime::Buffer<float, 2> mean_in(15, 15);
  mean(lab_in, mean_in);

  Halide::Runtime::Buffer<float, 2> kernel = MakeGaussianKernel();
  Halide::Runtime::Buffer<float, 2> variance_out(15, 15);
  variance(lab_in, mean_in, kernel, variance_out);

  Clamped in(lab_in.begin(), 15, 15);
  Clamped mean(mean_in.begin(), 15, 15);

  for (int i = 0; i < 15; ++i) {
    for (int j = 0; j < 15; ++j) {
      EXPECT_NEAR(
          ComputeWeightedVariance(in, mean, kernel.begin(), j, i),
          variance_out.begin()[(i * 15) + j], 0.01);
    }
  }
}

}  // namespace vcsmc
