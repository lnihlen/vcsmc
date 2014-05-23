#ifndef SRC_CL_BUFFER_IMPL_H_
#define SRC_CL_BUFFER_IMPL_H_

#include <OpenCL/OpenCL.h>

#include "cl_buffer.h"

namespace vcsmc {

class CLBufferImpl : public CLBuffer {
 public:
  CLBufferImpl();
  ~CLBufferImpl();

  bool Setup(size_t size, cl_context context);

  // CLBuffer
  virtual bool EnqueueCopyToDevice(
      CLCommandQueue* queue, const void* bytes) override;
  virtual bool EnqueueFill(
      CLCommandQueue* queue, const void* pattern, size_t pattern_size) override;
  virtual bool EnqueueCopyFromDevice(
      CLCommandQueue* queue, void* bytes) override;

  const cl_mem get() const { return mem_; }

 private:
  size_t size_;
  cl_mem mem_;
};

}  // namespace vcsmc

#endif  // SRC_CL_BUFFER_IMPL_H_
