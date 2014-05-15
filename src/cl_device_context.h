#ifndef SRC_CL_DEVICE_CONTEXT_H_
#define SRC_CL_DEVICE_CONTEXT_H_

#include <future>
#include <string>

#include "types.h"

namespace vcsmc {

class CLCommandQueue {
 public:
  // Blocks thread until queue is empty of all work items.
  void Finish();

 private:
  friend class CLDeviceContext;
  struct CLCommandQueueImpl;
  CLCommandQueue(std::unique_ptr<CLCommandQueueImpl> impl);
  std::unique_ptr<CLCommandQueueImpl> impl_;
};

class CLKernel {
 public:
  bool SetArgument(uint32 index, size_t size, const uint8* arg);
  bool Enqueue(CLCommandQueue* queue);
  //  bool Wait();

 private:
  friend class CLDeviceContext;
  struct CLKernelImpl;
  CLKernel(std::unique_ptr<CLKernelImpl> impl);
  std::unique_ptr<CLKernelImpl> impl_;
};

class CLBuffer {
 public:
  bool EnqueueCopyToDevice(CLCommandQueue* queue, const void* bytes);
  std::future<std::unique_ptr<uint8>> EnqueueCopyFromDevice(
      CLCommandQueue* queue);

 private:
  friend class CLDeviceContext;
  struct CLInputBufferImpl;
  CLInputBuffer(std::unique_ptr<CLInputBufferImpl> impl);
  std::unique_ptr<CLInputBufferImpl> impl_;
};

// Singleton class to represent single ownership of a global OpenCL device.
// Able to construct CLKernels, CLInputs and CLOutputs.
class CLDeviceContext {
  // Initialize singleton instance, returns false on failure.
  static bool Setup();
  static void Teardown();

  static std::unique_ptr<CLCommandQueue> MakeCommandQueue();
  static std::unique_ptr<CLKernel> MakeKernel(const std::string& kernel_name);
  static std::unique_ptr<CLInputBuffer> MakeBuffer(size_t size);

 private:
  friend class CLKernel;
  static CLDeviceContext* instance_;

  struct CLDeviceContextImpl;
  std::unique_ptr<CLDeviceContextImpl> impl_;
};

}  // namespace vcsmc

#endif
