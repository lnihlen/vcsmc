#ifndef SRC_CL_BUFFER_H_
#define SRC_CL_BUFFER_H_

namespace vcsmc {

class CLCommandQueue;

class CLBuffer {
 public:
  virtual bool EnqueueCopyToDevice(
      CLCommandQueue* queue, const void* data) = 0;
  // call queue->Finish() after issuing this before accessing data!
  virtual bool EnqueueCopyFromDevice(
      CLCommandQueue* queue, void* data) = 0;
};

}  // namespace vcsmc

#endif  // SRC_CL_BUFFER_H_
