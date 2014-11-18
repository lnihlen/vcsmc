#ifndef SRC_CL_COMMAND_QUEUE_H_
#define SRC_CL_COMMAND_QUEUE_H_

namespace vcsmc {

// Pure virtual class that defines an OpenCL Command Queue object. For code see
// CLCommandQueueImpl.
class CLCommandQueue {
 public:
  // Blocks thread until queue is empty of all work items.
  virtual void Finish() = 0;

  // Enqueues a barrier which won't allow the queue to proceed until all
  // previous commands have completed.
  virtual void EnqueueBarrier() = 0;

  virtual ~CLCommandQueue() {}
};

}  // namespace vcsmc

#endif  // SRC_CL_COMMAND_QUEUE_H_
