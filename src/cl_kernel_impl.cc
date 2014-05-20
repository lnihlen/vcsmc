#include "cl_kernel_impl.h"

#include "cl_buffer_impl.h"
#include "cl_command_queue_impl.h"
#include "cl_image_image.h"

namespace vcsmc {

CLKernelImpl::CLKernelImpl() : work_group_size_(0) {
}

CLKernelImpl::~CLKernelImpl() {
  if (kernel_) clReleaseKernel(kernel_);
}

bool CLKernelImpl::Setup(cl_program program,
                         const std::string& name,
                         cl_context context,
                         cl_device_id device_id) {
  int result = 0;
  kernel_ = clCreateKernel(program, name.c_str(), &result);
  if (!kernel_ || result != CL_SUCCESS)
    return false;

  result = clGetKernelWorkGroupInfo(kernel_,
                                    device_id,
                                    CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(work_group_size_),
                                    &work_group_size_,
                                    NULL);
  if (result != CL_SUCCESS)
    return false;

  event_ = clCreateUserEvent(context, &result);
  return (event_ && result == CL_SUCCESS);
}

size_t CLKernelImpl::WorkGroupSize() {
  return work_group_size_;
}

bool CLKernelImpl::SetByteArgument(
    uint32 index, size_t size, const uint8* arg) {
  int result = clSetKernelArg(kernel_, index, size, arg);
  return result == CL_SUCCESS;
}

bool CLKernelImpl::SetBufferArgument(uint32 index, const CLBuffer* buffer) {
  CLBuffferImpl* buffer_impl = static_cast<CLBufferImpl*>(buffer);
  int result = clSetKernelArg(
      kernel_, index, sizeof(cl_mem), buffer_impl->get());
  return result == CL_SUCCESS;
}

bool CLKernelImpl::SetImageArugment(uint32 index, const CLImage* image) {
  CLImageImpl* image_impl = static_cast<CLImageImpl*>(image);
  int result = clSetKernelArg(
      kernel_, index, sizeof(cl_mem), image_impl->get());
}

bool CLKernelImpl::Enqueue(CLCommandQueue* queue) {
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  size_t global_size = work_group_size_;
  int result = clEnqueueNDRangeKernel(queue_impl->get(),
                                      kernel_,
                                      1,
                                      NULL,
                                      &global_size,
                                      &work_group_size_,
                                      0,
                                      NULL,
                                      NULL);
  return (result == CL_SUCCESS);
}

}  // namespace vcsmc