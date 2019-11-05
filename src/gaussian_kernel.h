#ifndef SRC_GAUSSIAN_KERNEL_H_
#define SRC_GAUSSIAN_KERNEL_H_

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#include "Halide.h"
#pragma clang diagnostic pop

namespace vcsmc {

Halide::Runtime::Buffer<float, 2> MakeGaussianKernel();

}  // namespace vcsmc

#endif  // SRC_GAUSSIAN_KERNEL_H_
