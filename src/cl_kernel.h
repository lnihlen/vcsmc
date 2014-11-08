#ifndef SRC_CL_KERNEL_H_
#define SRC_CL_KERNEL_H_

#include "types.h"

namespace vcsmc {

class CLBuffer;
class CLCommandQueue;
class CLImage;

// Pure virtual class that defines an OpenCL Kernel object. For code see
// CLKernelImpl
class CLKernel {
 public:
  virtual size_t WorkGroupSize() = 0;
  virtual bool SetByteArgument(uint32 index, size_t size, const void* arg) = 0;
  virtual bool SetBufferArgument(uint32 index, const CLBuffer* buffer) = 0;
  virtual bool SetImageArgument(uint32 index, const CLImage* image) = 0;
  virtual bool Enqueue(CLCommandQueue* queue) = 0;
  virtual bool EnqueueWithGroupSize(CLCommandQueue* queue,
      size_t group_size) = 0;
};

}  // namespace vcsmc

#endif  // SRC_CL_KERNEL_H_
