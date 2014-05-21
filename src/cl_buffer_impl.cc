#include "cl_buffer_impl.h"

#include <cassert>

#include "cl_command_queue_impl.h"

namespace vcsmc {

CLBufferImpl::CLBufferImpl() : size_(0) {
}

CLBufferImpl::~CLBufferImpl() {
  if (event_) clReleaseEvent(event_);
  if (mem_) clReleaseMemObject(mem_);
}

bool CLBufferImpl::Setup(size_t size, cl_context context) {
  size_ = size;
  mem_ = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
  if (!mem_)
    return false;

  int result = 0;
  event_ = clCreateUserEvent(context, &result);
  return (result == CL_SUCCESS && event_);
}

bool CLBufferImpl::EnqueueCopyToDevice(
    CLCommandQueue* queue, const uint8* bytes) {
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  assert(queue_impl);
  int result = clEnqueueWriteBuffer(queue_impl->get(),
                                    mem_,
                                    CL_TRUE,
                                    0,
                                    size_,
                                    bytes,
                                    0,
                                    NULL,
                                    NULL);
  return result == CL_SUCCESS;
}

bool CLBufferImpl::EnqueueCopyFromDevice(
    CLCommandQueue* queue, uint8 * bytes) {
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  assert(queue_impl);
  int result = clEnqueueReadBuffer(queue_impl->get(),
                                   mem_,
                                   CL_TRUE,
                                   0,
                                   size_,
                                   bytes,
                                   0,
                                   NULL,
                                   NULL);
  return result == CL_SUCCESS;
}

}  // namespace vcsmc
