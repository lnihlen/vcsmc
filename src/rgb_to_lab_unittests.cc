#include "rgb_to_lab.h"
#include <gtest/gtest.h>

#include <cstring>
#include <stdio.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#include "Halide.h"
#pragma clang diagnostic pop

#include "types.h"

namespace {

// Pull in the test data table, generated by tools/make_rgb_to_lab_test_data.py
#include "rgb_lab_test_colors.inc"

}

namespace vcsmc {

TEST(RgbToLabTest, TestTable) {
  // 3D planar byte image data, 4096 colors in a 64x64 flattened array.
  Halide::Runtime::Buffer<uint8, 3> rgb_input(64, 64, 3);
  // Halide not happy wrapping a buffer around a const array in memory, so
  // we copy data into the buffer from the array.
  std::memcpy(rgb_input.begin(), kTestRGBImage, sizeof(kTestRGBImage));

  Halide::Runtime::Buffer<float, 3> lab_output(64, 64, 3);

  rgb_to_lab(rgb_input, lab_output);

  const size_t kAOffset = 64 * 64;
  const size_t kBOffset = 64 * 64 * 2;

  // Compare computed results to precomputed table.
  for (size_t i = 0; i < 64*64; ++i) {
    EXPECT_NEAR(kTestLABImage[i],
                lab_output.begin()[i], 0.01);
    EXPECT_NEAR(kTestLABImage[i + kAOffset],
                lab_output.begin()[i + kAOffset], 0.01);
    EXPECT_NEAR(kTestLABImage[i + kBOffset],
                lab_output.begin()[i + kBOffset], 0.01);
  }
}

}  // namespace vcsmc

