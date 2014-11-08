#ifndef SRC_CL_KERNEL_IMPL_H_
#define SRC_CL_KERNEL_IMPL_H_

#include <string>

#include "cl_include.h"
#include "cl_kernel.h"

namespace vcsmc {

class CLBuffer;
class CLCommandQueue;

class CLKernelImpl : public CLKernel {
 public:
  CLKernelImpl();
  virtual ~CLKernelImpl();

  bool Setup(cl_program program,
             const std::string& name,
             cl_context context,
             cl_device_id device_id);

  // CLKernel
  virtual size_t WorkGroupSize() override;
  virtual bool SetByteArgument(
      uint32 index, size_t size, const void* arg) override;
  virtual bool SetBufferArgument(uint32 index, const CLBuffer* buffer) override;
  virtual bool SetImageArgument(uint32 index, const CLImage* image) override;
  virtual bool Enqueue(CLCommandQueue* queue) override;
  virtual bool EnqueueWithGroupSize(CLCommandQueue* queue,
      size_t group_size) override;

 private:
  size_t work_group_size_;
  cl_kernel kernel_;
};

}  // namespace vcsmc

#endif  // SRC_CL_KERNEL_IMPL_H_
