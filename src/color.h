#ifndef SRC_COLOR_H_
#define SRC_COLOR_H_

#include "types.h"

namespace vcsmc {

// Maximum Lab value in our Lab definition.
const double kMaxLab = 100.0;

// Given four bytes of rgba color pointed to by |rgba| stores four doubles of
// LabA converted color values at |laba|.
void RGBAToLabA(const uint8* rgba, double* laba);

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
double Ciede2k(const double* laba_1, const double* laba_2);

}  // namespace vcsmc

#endif  // SRC_COLOR_H_
