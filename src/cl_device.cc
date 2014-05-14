#include "cl_device.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <OpenCL/OpenCL.h>

#include "cl_kernel.h"
#include "cl_task.h"

namespace vcsmc {

static CLDevice::CLDevice* instance_ = NULL;

// Technically both a kernel and a program, we assume a 1:1 relationship between
// .cl files in the "kernels/" directory so lump them together, and choose the
// name "program" because "kernel" aleady has meaning within vcsmc.
struct CLProgram {
  cl_program program;

  ~CLProgram() {
    clReleaseProgram(program);
  };
};

struct CLEvent {
  cl_event event;

};

struct CLDevice::Impl {
  cl_device_id device_id;
  cl_context context;
  cl_command_queue command_queue;

  // thread waits on this for something to happen.
  std::condition_variable cv;
  std::unique_lock<std::mutex> cv_lock;
  std::mutex cv_mutex;

  std::mutex tasks_mutex;
  std::list<std::unique_ptr<CLTask>> tasks;

  std::thread thread;
  // Not protected by lock.
  bool should_quit;

  std::mutex state_mutex;
  CLState state;

  // Our cache of previously loaded and compiled programs.
  typedef std::unordered_map<std::string, std::unique_ptr<CLProgram>>
      ProgramMap;
  ProgramMap program_map;

  uint64 event_handles;
  typedef std::unordered_map<uint64, std::unique_ptr<CLProgram>> CallbackMap;
  CallbackMap callback_map;

  Impl()
      : cv_lock(cv_mutex),
        should_quit(false),
        state(NOT_STARTED),
        event_handles(0)  { }
};


// static
bool CLDevice::Initialize() {
  instance_ = new CLDevice;
  instance_->impl_->thead = std::thread(&CLDevice::RunLoop, instance_);
}

// static
bool CLDevice::Teardown() {
}

CLDevice::CLDevice() : impl_(new Impl) {
}

bool CLDevice::InitializeOnOwnThread() {
  int result = clGetDeviceIDs(
      NULL, CL_DEVICE_TYPE_GPU, 1, &impl_->device_id, NULL);
  if (result != CL_SUCCESS)
    return false;

  impl_->context = clCreateContext(
      0, 1, &impl_->device_id, NULL, NULL, &result);
  if (!context)
    return false;

  impl_->command_queue = clCreateCommandQueue(
      context, impl_->device_id, 0, &result);
  if (!impl_->command_queue)
    return false;

  return true;
}

void CLDevice::RunLoop() {
  SetState(INITIALIZING);

  if (!InitializeOnOnThread()) {
    SetState(FATAL_ERROR);
    return;
  }

  SetState(READY);

  // Main loop. Wait for task enqueue.
  while (!should_quit) {
    impl_->cv.wait();

    std::unique_ptr<Task> task;

    {
      std::lock_guard<std::mutex> lock(impl_->tasks_mutex);
      if (tasks.size()) {
        task = std::move(tasks.front());
        tasks.pop_front();
      }
    }

    if (!task)
      continue;

    ProgramMap::iterator pg = impl_->program_map.find(task->program_name());
    // Load and compile program if we have not yet done so.
    if (pg == impl_->program_map.end()) {
      if (!LoadProgram(task->program_name())) {
        task->OnFailure();
      }
      pg = impl_->program_map.find(task->program_name());
    }

    assert(pg != impl_->program_map.end());

    int result;

    // Make a kernel and connect the arguments.
    cl_kernel kernel = clCreateKernel(
        program_map.find((*pg)->program, task->program_name(), &result));
    if (result != CL_SUCCESS) {
      SetState(FATAL_ERROR);
      return;
    }

    size_t arg = 0
    for (size_t i = 0; i < task->inputs(); ++i) {
      result |= clSetKernelArg(
          kernel, arg++, task->input(i).first, task->input(i).second);
    }
    for (size_t i = 0; i < task->outputs(); ++i) {
      result |= clSetKernelArg(
          kernel, arg++, task->output(i).first, task->output(i).second);
    }
    if (result != CL_SUCCESS) {
      SetState(FATAL_ERROR);
      return;
    }

    // Create an event and connect it to our callback function.

    // Enqueue.
  }
}

void CLDevice::SetState(CLState state) {
  std::lock_guard<std::mutex> lock(impl_->state_mutex);
  impl_->state = state;
}

}  // namespace vcsmc
