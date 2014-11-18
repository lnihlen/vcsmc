#include "cl_buffer_impl.h"

#include <cassert>
#include <cstring>
#include <stdio.h>

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
  return mem_ != 0;
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

bool CLBufferImpl::EnqueueFill(
    CLCommandQueue* queue, const void* pattern, size_t pattern_size) {
  assert(size_ % pattern_size == 0);
  CLCommandQueueImpl* queue_impl = static_cast<CLCommandQueueImpl*>(queue);
  assert(queue_impl);

#if defined(NVIDIA_OPENCL_LAMENESS)
  assert(!fill_buffer_);
  fill_buffer_.reset(new uint8[size_]);
  uint8* buffer_ptr = fill_buffer_.get();
  size_t number_of_copies = size_ / pattern_size;
  for (size_t i = 0; i < number_of_copies; ++i) {
    std::memcpy(buffer_ptr, pattern, pattern_size);
    buffer_ptr += pattern_size;
  }
  int result = clEnqueueWriteBuffer(queue_impl->get(),
                                    mem_,
                                    CL_TRUE,
                                    0,
                                    size_,
                                    fill_buffer_.get(),
                                    0,
                                    NULL,
                                    NULL);
#else
  int result = clEnqueueFillBuffer(queue_impl->get(),
                                   mem_,
                                   pattern,
                                   pattern_size,
                                   0,
                                   size_,
                                   0,
                                   NULL,
                                   NULL);
#endif  // NVIDIA_OPENCL_LAMENESS
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
