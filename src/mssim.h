#ifndef SRC_MSSIM_H_
#define SRC_MSSIM_H_

#include "constants.h"
#include "types.h"

namespace vcsmc {

const size_t kWindowSize = 8;
const size_t kLBufferSize = kTargetFrameWidthPixels * kFrameHeightPixels;
const size_t kLBufferSizeBytes = sizeof(float) * kLBufferSize;
const size_t kLabaBufferSize = 4 * kLBufferSize;
const size_t kLabaBufferSizeBytes = sizeof(float) * kLabaBufferSize;

// Given input buffer of L*a*b* color, reduces the output to single normalized
// luminance value, normalized to [0,1].
__global__ void LabaToNormalizedL(const float4* laba_in,
                                  float* nl_out);

// Given two images in packed doubles with L*a*b* + alpha color format, and
// their dimensions, this algorithm will compute and return the Mean Structural
// Simularity between the two images, as described by the paper:
//
//   Z. Wang, L. Lu, and A. C. Bovik, “Video quality assessment based on
//   structural distortion measurement,” Signal Process.: Image Commun.,
//   vol. 19, no. 2, pp. 121–132, Feb. 2004
//
__global__ void ComputeMean(const float* nl_in,
                            float* mean_out);

__global__ void ComputeVariance(const float* nl_in,
                                const float* mean_in,
                                float* variance_out);

__global__ void ComputeCovariance(const float* nl_a_in,
                                  const float* mean_a_in,
                                  const float* nl_b_in,
                                  const float* mean_b_in,
                                  float* cov_ab_out);

__global__ void ComputeSSIM(const float* mean_a_in,
                            const float* variance_a_in,
                            const float* mean_b_in,
                            const float* variance_b_in,
                            const float* cov_ab_in,
                            float* ssim_out);

__global__ void ComputeBlockSum(const float* ssim_in, float* block_sum_out);

}  // namespace vcsmc

#endif  // SRC_MSSIM_H_
