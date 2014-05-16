#ifndef SRC_CL_COMMAND_QUEUE_IMPL_H_
#define SRC_CL_COMMAND_QUEUE_IMPL_H_

#include <OpenCL/OpenCL.h>

#include "cl_command_queue.h"

namespace vcsmc {

class CLCommandQueueImpl : public CLCommandQueue {
 public:
  CLCommandQueueImpl();
  virtual ~CLCommandQueueImpl();

  bool Setup(cl_context context, cl_device_id device_id);
  cl_command_queue get() { return command_queue_; }

  // CLCommandQueue
  virtual void Finish() override;

 private:
  cl_command_queue command_queue_;
};

}  // namespace vcsmc

#endif  // SRC_CL_COMMAND_QUEUE_IMPL_H_
