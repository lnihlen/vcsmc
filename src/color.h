#ifndef SRC_COLOR_H_
#define SRC_COLOR_H_

#include "types.h"

namespace vcsmc {

// Given four bytes of rgba color pointed to by |rgba| stores four floats of
// (Y, U, V, 1.0)  converted color values at |yuv|. The normalization maps the
// maximum range of YUV colors as converted from RGB to the range [0..1] for
// each value.
void RgbaToNormalizedYuv(const uint8* rgba, float* nyuv);

}  // namespace vcsmc

#endif  // SRC_COLOR_H_
