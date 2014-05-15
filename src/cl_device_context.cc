#include "cl_device_context.h"

#include <OpenCL/OpenCL.h>

namespace vcsmc {


// CLCommandQueue -------------------------------------------------------------

struct CLCommandQueueImpl {
  cl_command_queue command_queue;

  ~CLCommandQueueImpl() {
    clReleaseCommandQueue(command_queue);
  }
};

void CLCommandQueue::Finish() {
  clFinish(impl_->command_queue);
}


// CLKernel -------------------------------------------------------------------

struct CLKernelImpl {
  cl_kernel kernel;
  cl_event event;

  ~CLKernelImpl {
    clReleaseEvent(event);
    clReleaseKernel(kernel);
  }
};

bool CLKernel::SetArgument(uint32 index, size_t size, const uint8* arg) {
  int result = clSetKernelArg(impl_->kernel, index, size, arg);
  return (result == CL_SUCCESS);
}

bool CLKernel::Enqueue(CLCommandQueue* queue) {

}

}  // namespace vcsmc
