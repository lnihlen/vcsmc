#ifndef SRC_CL_BUFFER_H_
#define SRC_CL_BUFFER_H_

#include "types.h"

namespace vcsmc {

class CLCommandQueue;

class CLBuffer {
 public:
  virtual bool EnqueueCopyToDevice(
      CLCommandQueue* queue, const uint8* bytes) = 0;
  // call queue->Finish() after issuing this before accessing data!
  virtual bool EnqueueCopyFromDevice(
      CLCommandQueue* queue, uint8* bytes) = 0;
};

}  // namespace vcsmc

#endif  // SRC_CL_BUFFER_H_
