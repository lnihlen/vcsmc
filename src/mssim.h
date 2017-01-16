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

}  // namespace vcsmc

#endif  // SRC_MSSIM_H_
