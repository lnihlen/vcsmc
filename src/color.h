#ifndef SRC_COLOR_H_
#define SRC_COLOR_H_

#include "types.h"

namespace vcsmc {

// Maximum Lab value in our Lab definition.
const float kMaxLab = 100.0;

// Given four bytes of rgba color pointed to by |rgba| stores three floats of
// L*a*b* converted color values at |lab|.
void RgbaToLaba(const uint8* rgba, float* lab);

}  // namespace vcsmc

#endif  // SRC_COLOR_H_
