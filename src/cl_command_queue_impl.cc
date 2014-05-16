#include "cl_command_queue_impl.h"

namespace vcsmc {

CLCommandQueueImpl::CLCommandQueueImpl() {
}

CLCommandQueueImpl::~CLCommandQueueImpl() {
  clReleaseCommandQueue(command_queue_);
}

bool CLCommandQueueImpl::Setup(cl_context context, cl_device_id device_id) {
  int result;
  command_queue_ = clCreateCommandQueue(context, device_id, 0, &result);
  return (command_queue_ && result == CL_SUCCESS);
}

void CLCommandQueueImpl::Finish() {
  clFinish(command_queue_);
}

}  // namespace vcsmc
