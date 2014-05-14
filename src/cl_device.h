#ifndef SRC_CL_DEVICE_H_
#define SRC_CL_DEVICE_H_

#include <string>

namespace vcsmc {

class CLTask;

// Simplisticly represents an OpenCL Device, Context, and Command Queue.
class CLDevice {
 public:

  // Spawns the thread to actually setup OpenCL.
  static void Initialize();
  static void Terminate();

  enum CLState {
    NOT_STARTED,
    INITIALIZING,
    READY,
    TERMINATING,
    STOPPED,
    FATAL_ERROR
  };

  // Takes ownership of task. Guaranteed FIFO order
  static void Enqueue(std::unique_ptr<CLTask> task);

 private:
  CLDevice();
  bool InitializeOnOwnThread();
  void RunLoop();
  void SetState(CLState state);

  static CLDevice* instance_;

  bool LoadProgram(const std::string& name);

  // Keep the OpenCL state block in the .cc to avoid polluting the header.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vcsmc

#endif
