#ifndef SRC_CL_BUFFER_IMPL_H_
#define SRC_CL_BUFFER_IMPL_H_

#if defined(NVIDIA_OPENCL_LAMENESS)
#include <memory>
#endif

#include "cl_buffer.h"
#include "cl_include.h"
#include "types.h"

namespace vcsmc {

class CLBufferImpl : public CLBuffer {
 public:
  CLBufferImpl();
  virtual ~CLBufferImpl();

  bool Setup(size_t size, cl_context context);

  // CLBuffer
  virtual bool EnqueueCopyToDevice(
      CLCommandQueue* queue, const void* bytes) override;
  virtual bool EnqueueFill(
      CLCommandQueue* queue, const void* pattern, size_t pattern_size) override;
  virtual bool EnqueueCopyFromDevice(
      CLCommandQueue* queue, void* bytes) override;

  cl_mem get() const { return mem_; }

 private:
  size_t size_;
  cl_mem mem_;

#if defined(NVIDIA_OPENCL_LAMENESS)
  std::unique_ptr<uint8[]> fill_buffer_;
#endif
};

}  // namespace vcsmc

#endif  // SRC_CL_BUFFER_IMPL_H_
