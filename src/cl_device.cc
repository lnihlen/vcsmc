#include "cl_device.h"

#include <iostream>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <OpenCL/OpenCL.h>

#include "cl_task.h"

namespace vcsmc {

struct CLDevice::Impl {
  cl_device_id device_id;
  cl_context context;
  cl_command_queue command_queue;

  // thread waits on this for something to happen.
  std::condition_variable cv;
  std::unique_lock<std::mutex> cv_lock;
  std::mutex cv_mutex;

  // Protects tasks
  std::mutex tasks_mutex;
  std::list<std::unique_ptr<CLTask>> tasks;

  std::thread thread;
  // Not protected by lock.
  bool should_quit;

  Impl()
      : cv_lock(cv_mutex),
        should_quit(false) { }
};

static CLDevice::CLDevice* instance_ = NULL;

// static
bool CLDevice::Setup() {
  instance_ = new CLDevice;
  instance_->impl_->thead = std::thread(&CLDevice::RunLoop, instance_);
}

// static
bool CLDevice::Teardown() {
}

CLDevice::CLDevice() : impl_(new Impl) {
}

void CLDevice::RunLoop() {
  int result = clGetDeviceIDs(
      NULL, CL_DEVICE_TYPE_GPU, 1, &impl_->device_id, NULL);
  if (result != CL_SUCCESS) {
    std::cerr << "failed to create device group: " << result << std::endl;
    return;
  }

  impl_->context = clCreateContext(
      0, 1, &impl_->device_id, NULL, NULL, &result);
  if (!context) {
    std::cerr << "failed to create compute context: " << result <<  std::endl;
    return;
  }

  impl_->command_queue = clCreateCommandQueue(
      context, impl_->device_id, 0, &result);
  if (!impl_->command_queue) {
    std::cerr << "failed to create command queue: " << result << std::endl;
    return;
  }

  // Main loop. Wait for task enqueue.
  while (!should_quit) {
    impl_->cv.wait();

    std::unique_ptr<Task> current_task;

    {
      std::lock_guard<std::mutex> lock(tasks_mutex);
      if (tasks.size()) {
        current_task = std::move(tasks.front());
        tasks.pop_front();
      }
    }

    if (!current_task)
      continue;
  }
}

}  // namespace vcsmc
