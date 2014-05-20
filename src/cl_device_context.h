#ifndef SRC_CL_DEVICE_CONTEXT_H_
#define SRC_CL_DEVICE_CONTEXT_H_

#include <string>

#include "types.h"

namespace vcsmc {

class CLCommandQueue;
class CLImage;
class CLKernel;
class CLBuffer;
class Image;
class PixelStrip;

// Singleton class to represent single ownership of a global OpenCL device.
// Constructs all other CL objects.
class CLDeviceContext {
 public:
  // Initialize singleton instance, returns false on failure.
  static bool Setup();
  static void Teardown();

  enum Kernels : size_t {
    kCiede2k = 0,
    kRGBToLab = 1,
    kKernelsCount = 2
  };

  static std::unique_ptr<CLBuffer> MakeBuffer(size_t size);
  static std::unique_ptr<CLCommandQueue> MakeCommandQueue();
  static std::unique_ptr<CLImage> MakeImage(const Image* image);
  static std::unique_ptr<CLImage> MakeImageFromStrip(const PixelStrip* strip);
  static std::unique_ptr<CLKernel> MakeKernel(Kernels kernel);

 private:
  static CLDeviceContext* instance_;

  CLDeviceContext();
  ~CLDeviceContext();

  bool DoSetup();
  bool LoadAndBuildProgram(Kernels kernel);
  const char* KernelName(Kernels kernel);
  std::unique_ptr<CLBuffer> DoMakeBuffer(size_t size);
  std::unique_ptr<CLCommandQueue> DoMakeCommandQueue();
  std::unique_ptr<CLImage> DoMakeImage(const Image* image);
  std::unique_ptr<CLImage> DoMakeImageFromStrip(const PixelStrip* strip);
  std::unique_ptr<CLKernel> DoMakeKernel(Kernels kernel);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vcsmc

#endif
