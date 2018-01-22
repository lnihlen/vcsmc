#ifndef SRC_CIEDE_2K_H_
#define SRC_CIEDE_2K_H_

namespace vcsmc {

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
__global__ void Ciede2k(const float4* laba_a_in,
                        const float4* laba_b_in,
                        float* ciede_2k_out);

}  // namespace vcsmc

#endif  // SRC_CIEDE_2K_H_
