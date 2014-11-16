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
  virtual bool Enqueue(CLCommandQueue* queue, size_t work_size) = 0;
  virtual bool Enqueue2D(CLCommandQueue* queue, size_t work_width,
      size_t work_height) = 0;

  // Returns local memory consumed by the kernel, including any local memory
  // area arguments whose size has been specified.
  virtual uint64 LocalMemoryUsed() = 0;
};

}  // namespace vcsmc

#endif  // SRC_CL_KERNEL_H_
