#ifndef SRC_MSSIM_H_
#define SRC_MSSIM_H_

#include "constants.h"
#include "types.h"

namespace vcsmc {

const size_t kWindowSize = 8;
const size_t kLabaBufferSize = 4 *
    (kTargetFrameWidthPixels + kWindowSize) *
    (kFrameHeightPixels + kWindowSize);
const size_t kLabaBufferSizeBytes = sizeof(float) * kLabaBufferSize;


// Packed images come in as 1 Atari color byte representing 2 horizontal pixel
// colors. It is assumed that the blocks are 16 pixels wide in double
// horizontal screen resolution, so that would mean 8 bytes of Atari colors.
// This function wants to detect if any pixel relevant to the 16x16 block has
// changed, which actually means it needs to inspect a 24x24 block, to include
// the window of pixels encompassed by the pixels on the right and bottom edge
// of the current block. For speed of comparison we repeat the 8 pixels (so 4
// bytes) of data outside of the block on the right, and then for alignment to
// 16 bytes instead of 12 (for 24 pixels) we pad on 4 bytes of zero, which also
// serves as zero padding on the right side of the image to allow for loop
// unrolling.
__global__ void FindChangedBlocks(const uint4* parent_packed_image_in,
                                  const uint4* child_packed_image_in,
                                  uchar1* block_mask_out);

__global__ void UnpackToPaddedNyuv(const uchar1* parent_packed_image_in,
                                  const float4* laba_table,
                                  const uchar1* block_mask_in,  // optional
                                  float4* laba_out);

// Given two images in packed doubles with L*a*b* + alpha color format, and
// their dimensions, this algorithm will compute and return the Mean Structural
// Simularity between the two images, as described by the paper:
//
//   Z. Wang, L. Lu, and A. C. Bovik, “Video quality assessment based on
//   structural distortion measurement,” Signal Process.: Image Commun.,
//   vol. 19, no. 2, pp. 121–132, Feb. 2004
//
__global__ void ComputeLocalMean(const float4* laba_in,
//                                 const uchar1* block_mask_in, // optional
                                 float4* mean_out);

__global__ void ComputeLocalStdDevSquared(const float4* laba_in,
                                          const float4* mean_in,
//                                          const uchar1* block_mask_in, // opt
                                          float4* stddevsq_out);

__global__ void ComputeLocalCovariance(const float4* laba_a_in,
                                       const float4* mean_a_in,
                                       const float4* laba_b_in,
                                       const float4* mean_b_in,
//                                       const uchar1* block_mask_in, // optional
                                       float4* cov_ab_out);

__global__ void ComputeSSIM(const float4* mean_a_in,
                            const float4* stddevsq_a_in,
                            const float4* mean_b_in,
                            const float4* stddevsq_b_in,
                            const float4* cov_ab_in,
//                            const uchar1* block_mask_in, // optional
                            float* ssim_out);

__global__ void ComputeBlockSum(const float* ssim_in, float* block_sum_out);

}  // namespace vcsmc

#endif  // SRC_MSSIM_H_
