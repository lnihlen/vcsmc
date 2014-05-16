#ifndef SRC_CL_BUFFER_H_
#define SRC_CL_BUFFER_H_

#include <future>

#include "types.h"

namespace vcsmc {

class CLCommandQueue;

class CLBuffer {
 public:
  virtual bool EnqueueCopyToDevice(
      CLCommandQueue* queue, const uint8* bytes) = 0;
  virtual std::future<std::unique_ptr<uint8>> EnqueueCopyFromDevice(
      CLCommandQueue* queue) = 0;
};

}  // namespace vcsmc

#endif  // SRC_CL_BUFFER_H_
