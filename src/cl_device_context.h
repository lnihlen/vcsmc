#ifndef SRC_CL_DEVICE_CONTEXT_H_
#define SRC_CL_DEVICE_CONTEXT_H_

#include <string>

#include "types.h"

namespace vcsmc {

class CLCommandQueue;
class CLKernel;
class CLBuffer;

// Singleton class to represent single ownership of a global OpenCL device.
// Constructs all other CL objects.
class CLDeviceContext {
  // Initialize singleton instance, returns false on failure.
  static bool Setup();
  static void Teardown();

  enum Kernels : size_t {
    kCiede2k = 0,
    kRGBToLab = 1,
    kKernelsCount = 2
  };

  static std::unique_ptr<CLCommandQueue> MakeCommandQueue();
  static std::unique_ptr<CLKernel> MakeKernel(Kernels kernel);
  static std::unique_ptr<CLBuffer> MakeBuffer(size_t size);

 private:
  static CLDeviceContext* instance_;

  CLDeviceContext();
  ~CLDeviceContext();

  bool DoSetup();
  bool LoadAndBuildProgram(Kernels kernel);
  const char* KernelName(Kernels kernel);
  std::unique_ptr<CLCommandQueue> DoMakeCommandQueue();
  std::unique_ptr<CLKernel> DoMakeKernel(Kernels kernel);
  std::unique_ptr<CLBuffer> DoMakeBuffer(size_t size);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vcsmc

#endif
