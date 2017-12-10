#ifndef SRC_COLOR_H_
#define SRC_COLOR_H_

#include "types.h"

namespace vcsmc {

// Maximum Lab value in our Lab definition.
const float kMaxLab = 100.0;

// Given four bytes of rgba color pointed to by |rgba| stores three floats of
// L*a*b* converted color values at |lab|.
void RGBAToLab(const uint8* rgba, float* lab);

// Given four bytes of rgba color pointed to by |rgba| stores four floats of
// (Y, U, V, 1.0)  converted color values at |yuv|. The normalization maps the
// maximum range of YUV colors as converted from RGB to the range [0..1] for
// each value.
void RGBAToNormalizedYuv(const uint8* rgba, float* nyuv);

// Given input colors in L*a*b* color space returns the CIEDE2000 delta E value,
// a nonlinear color distance based on the human vision response.
//
// This is by no means an official implementation of this algorithm.
//
// This implementation is with thanks to:
//
// "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary
// Test Data, and Mathematical Observations,", G. Sharma, W. Wu, E. N. Dalal,
// Color Research and Application, vol. 30. No. 1, February 2005.
//
// Note that CIEDE2000 is discontinuous so is not useful for gradient
// descent algorithms.
float Ciede2k(const float* laba_1, const float* laba_2);

}  // namespace vcsmc

#endif  // SRC_COLOR_H_
