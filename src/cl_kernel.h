#ifndef SRC_CL_KERNEL_H_
#define SRC_CL_KERNEL_H_

#include "types.h"

namespace vcsmc {

class CLCommandQueue;

// Pure virtual class that defines an OpenCL Kernel object. For code see
// CLKernelImpl
class CLKernel {
 public:
  virtual size_t WorkGroupSize() = 0;
  virtual bool SetArgument(uint32 index, size_t size, const uint8* arg) = 0;
  virtual bool Enqueue(CLCommandQueue* queue) = 0;
};

}  // namespace vcsmc

#endif  // SRC_CL_KERNEL_H_
