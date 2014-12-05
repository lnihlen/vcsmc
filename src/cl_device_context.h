#ifndef SRC_CL_DEVICE_CONTEXT_H_
#define SRC_CL_DEVICE_CONTEXT_H_

#include <memory>
#include <string>

#include "types.h"
#include "cl_program.h"

namespace vcsmc {

class CLBuffer;
class CLCommandQueue;
class CLImage;
class CLKernel;
class CLProgram;
class Image;

// Singleton class to represent single ownership of a global OpenCL device.
// Constructs all other CL objects.
class CLDeviceContext {
 public:
  // Initialize singleton instance, returns false on failure.
  static bool Setup();
  static void Teardown();

  static std::unique_ptr<CLBuffer> MakeBuffer(size_t size);
  static std::unique_ptr<CLCommandQueue> MakeCommandQueue();
  static std::unique_ptr<CLImage> MakeImage(const Image* image);
  static std::unique_ptr<CLKernel> MakeKernel(CLProgram::Programs program);
  static uint64 LocalMemorySize();

 private:
  static CLDeviceContext* instance_;

  CLDeviceContext();
  ~CLDeviceContext();

  bool DoSetup();
  bool BuildProgram(CLProgram::Programs program);
  std::unique_ptr<CLBuffer> DoMakeBuffer(size_t size);
  std::unique_ptr<CLCommandQueue> DoMakeCommandQueue();
  std::unique_ptr<CLImage> DoMakeImage(const Image* image);
  std::unique_ptr<CLKernel> DoMakeKernel(CLProgram::Programs program);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vcsmc

#endif
