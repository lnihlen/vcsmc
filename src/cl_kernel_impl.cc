#include "cl_kernel_impl.h"

#include <cassert>

#include "cl_buffer_impl.h"
#include "cl_command_queue_impl.h"
#include "cl_image_impl.h"

namespace vcsmc {

CLKernelImpl::CLKernelImpl()
    : work_group_size_(0),
      local_memory_used_(0) {
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

  result = clGetKernelWorkGroupInfo(kernel_,
                                    device_id,
                                    CL_KERNEL_LOCAL_MEM_SIZE,
                                    sizeof(local_memory_used_),
                                    &local_memory_used_,
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

bool CLKernelImpl::Enqueue(CLCommandQueue* queue, size_t work_size) {
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  assert(queue_impl);
  int result = clEnqueueNDRangeKernel(queue_impl->get(),
                                      kernel_,
                                      1,
                                      NULL,
                                      &work_size,
                                      NULL,
                                      0,
                                      NULL,
                                      NULL);
  return (result == CL_SUCCESS);
}

bool CLKernelImpl::Enqueue2D(CLCommandQueue* queue, size_t work_width,
      size_t work_height) {
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  assert(queue_impl);
  size_t work_dim[2] = { work_width, work_height };
  int result = clEnqueueNDRangeKernel(queue_impl->get(),
                                      kernel_,
                                      2,
                                      NULL,
                                      work_dim,
                                      NULL,
                                      0,
                                      NULL,
                                      NULL);
  return (result == CL_SUCCESS);
}

bool CLKernelImpl::EnqueueWithOffset(CLCommandQueue* queue, size_t dimension,
      const size_t* sizes, const size_t* offsets) {
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  assert(queue_impl);
  int result = clEnqueueNDRangeKernel(queue_impl->get(),
                                      kernel_,
                                      dimension,
                                      offsets,
                                      sizes,
                                      NULL,
                                      0,
                                      NULL,
                                      NULL);
  return (result == CL_SUCCESS);
}

uint64 CLKernelImpl::LocalMemoryUsed() {
  return local_memory_used_;
}

}  // namespace vcsmc
