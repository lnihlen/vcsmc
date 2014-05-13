#ifndef SRC_CL_DEVICE_H_
#define SRC_CL_DEVICE_H_

namespace vcsmc {

// Simplisticly represents an OpenCL Device, Context, and Command Queue.
class CLDevice {
 public:
  // Spawns the thread to actually setup OpenCL.
  static void Setup();
  static void Teardown();

  class Task {
   public:
    Task();

   private:
  };

 private:
  CLDevice();
  void RunLoop();

  static CLDevice* instance_;
  // Keep the OpenCL state block in the .cc to avoid polluting the header.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vcsmc

#endif
