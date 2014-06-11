#ifndef SRC_CL_BUFFER_H_
#define SRC_CL_BUFFER_H_

#include "types.h"

namespace vcsmc {

class CLCommandQueue;

class CLBuffer {
 public:
  virtual bool EnqueueCopyToDevice(
      CLCommandQueue* queue, const void* data) = 0;
  virtual bool EnqueueFill(
      CLCommandQueue* queue, const void* pattern, size_t pattern_size) = 0;
  // Call queue->Finish() after issuing this before accessing data!
  virtual bool EnqueueCopyFromDevice(
      CLCommandQueue* queue, void* data) = 0;
};

}  // namespace vcsmc

#endif  // SRC_CL_BUFFER_H_
