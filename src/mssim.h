#ifndef SRC_MSSIM_H_
#define SRC_MSSIM_H_

#include "types.h"

namespace vcsmc {

// Given two images in packed doubles with L*a*b* + alpha color format, and
// their dimensions, this algorithm will compute and return the Mean Structural
// Simularity between the two images, as described by the paper:
//
//   Z. Wang, L. Lu, and A. C. Bovik, “Video quality assessment based on
//   structural distortion measurement,” Signal Process.: Image Commun.,
//   vol. 19, no. 2, pp. 121–132, Feb. 2004
//
double Mssim(const double* lab_a,
             const double* lab_b,
             uint32 image_width,
             uint32 image_height);

__global__ void ComputeLocalMean(const float3 lab_in[width, height],
                                 float3[width][height] mean_out);

__global__ void ComputeLocalStdDevSquared(const float3 lab_in[width, height],
                                          const float3 mean_in[width, height],
                                          float3 stddevsq_out[width, height]);

__global__ void ComputeLocalCovariance(const float3 lab_a_in[width, height],
                                       const float3 mean_a_in[width, height],
                                       const float3 lab_b_in[width, height],
                                       const float3 mean_b_in[width, height],
                                       float3 cov_ab_out[width, height]);

__global__ void ComputeSSIM(const float3 mean_a_in[width, height],
                            const float3 stddevsq_a_in[width, height],
                            const float3 mean_b_in[width, height],
                            const float3 stddevsq_b_in[width, height],
                            const float3 cov_ab_in[width, height],
                            float* ssim_out);

__global__ void ComputeBlockSum(const float* ssim_in, float* block_sum_out);

}  // namespace vcsmc

#endif  // SRC_MSSIM_H_
