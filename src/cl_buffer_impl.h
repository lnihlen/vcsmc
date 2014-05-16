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
      CLCommandQueue* queue, const uint8* bytes) override;
  virtual std::future<std::unique_ptr<uint8>> EnqueueCopyFromDevice(
      CLCommandQueue* queue) override;

 private:
  size_t size_;
  cl_event event_;
  cl_mem mem_;
  std::promise<std::unique_ptr<uint8>> promise_;
};

}  // namespace vcsmc

#endif  // SRC_CL_BUFFER_IMPL_H_
