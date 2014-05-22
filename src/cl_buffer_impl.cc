#include "cl_buffer_impl.h"

#include <cassert>

#include "cl_command_queue_impl.h"

namespace vcsmc {

CLBufferImpl::CLBufferImpl() : size_(0) {
}

CLBufferImpl::~CLBufferImpl() {
  if (mem_) clReleaseMemObject(mem_);
}

bool CLBufferImpl::Setup(size_t size, cl_context context) {
  size_ = size;
  mem_ = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
  return mem_;
}

bool CLBufferImpl::EnqueueCopyToDevice(
    CLCommandQueue* queue, const void* bytes) {
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
    CLCommandQueue* queue, void* bytes) {
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
