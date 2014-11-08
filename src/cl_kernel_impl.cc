#include "cl_kernel_impl.h"

#include <cassert>

#include "cl_buffer_impl.h"
#include "cl_command_queue_impl.h"
#include "cl_image_impl.h"

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
  return result == CL_SUCCESS;
}

size_t CLKernelImpl::WorkGroupSize() {
  return work_group_size_;
}

bool CLKernelImpl::SetByteArgument(
    uint32 index, size_t size, const void* arg) {
  int result = clSetKernelArg(kernel_, index, size, arg);
  return result == CL_SUCCESS;
}

bool CLKernelImpl::SetBufferArgument(uint32 index, const CLBuffer* buffer) {
  const CLBufferImpl* buffer_impl = static_cast<const CLBufferImpl*>(buffer);
  assert(buffer_impl);
  const cl_mem buffer_mem = buffer_impl->get();
  assert(buffer_mem);
  int result = clSetKernelArg(kernel_, index, sizeof(cl_mem), &buffer_mem);
  return result == CL_SUCCESS;
}

bool CLKernelImpl::SetImageArgument(uint32 index, const CLImage* image) {
  const CLImageImpl* image_impl = static_cast<const CLImageImpl*>(image);
  assert(image_impl);
  const cl_mem image_mem = image_impl->get();
  assert(image_mem);
  int result = clSetKernelArg(kernel_, index, sizeof(cl_mem), &image_mem);
  return result == CL_SUCCESS;
}

bool CLKernelImpl::Enqueue(CLCommandQueue* queue) {
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  assert(queue_impl);
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

bool CLKernelImpl::EnqueueWithGroupSize(CLCommandQueue* queue,
    size_t group_size) {
  // Could break this out into local work groups versus global work groups but
  // for now it just is a failure.
  assert(group_size < work_group_size_);
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  assert(queue_impl);
  size_t global_size = group_size;
  size_t work_group_size = group_size;
  int result = clEnqueueNDRangeKernel(queue_impl->get(),
                                      kernel_,
                                      1,
                                      NULL,
                                      &global_size,
                                      &work_group_size,
                                      0,
                                      NULL,
                                      NULL);
  return (result == CL_SUCCESS);
}

}  // namespace vcsmc
