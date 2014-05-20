#include "cl_device_context.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <OpenCL/OpenCL.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cl_buffer_impl.h"
#include "cl_command_queue_impl.h"
#include "cl_image_impl.h"
#include "cl_kernel_impl.h"
#include "image.h"
#include "pixel_strip.h"

namespace vcsmc {

CLDeviceContext* CLDeviceContext::instance_ = NULL;

struct CLDeviceContext::Impl {
  cl_device_id device_id;
  cl_context context;

  // Our cache of previously loaded and compiled programs.
  cl_program programs[Kernels::kKernelsCount];
};

// static
bool CLDeviceContext::Setup() {
  instance_ = new CLDeviceContext();
  return instance_->DoSetup();
}

// static
void CLDeviceContext::Teardown() {
  delete instance_;
  instance_ = NULL;
}

// static
std::unique_ptr<CLBuffer> CLDeviceContext::MakeBuffer(size_t size) {
  return instance_->DoMakeBuffer(size);
}

// static
std::unique_ptr<CLCommandQueue> CLDeviceContext::MakeCommandQueue() {
  return instance_->DoMakeCommandQueue();
}

// static
std::unique_ptr<CLImage> CLDeviceContext::MakeImage(const Image* image) {
  return instance_->DoMakeImage(image);
}

// static
std::unique_ptr<CLImage> CLDeviceContext::MakeImageFromStrip(
    const PixelStrip* strip) {
  return instance_->DoMakeImageFromStrip(strip);
}

// static
std::unique_ptr<CLKernel> CLDeviceContext::MakeKernel(Kernels kernel) {
  return instance_->DoMakeKernel(kernel);
}

CLDeviceContext::CLDeviceContext() : impl_(new Impl) {
}

CLDeviceContext::~CLDeviceContext() {
  for (size_t i = 0; i < Kernels::kKernelsCount; ++i) {
    clReleaseProgram(impl_->programs[i]);
  }
  clReleaseContext(impl_->context);
}

bool CLDeviceContext::DoSetup() {
  int result = clGetDeviceIDs(
      NULL, CL_DEVICE_TYPE_GPU, 1, &impl_->device_id, NULL);
  if (result != CL_SUCCESS)
    return false;

  impl_->context = clCreateContext(
      0, 1, &impl_->device_id, NULL, NULL, &result);
  if (!impl_->context)
    return false;

  // Load all program by hand, during the setup function. This allows easy error
  // reporting and program termination if there are problems, saves me having to
  // design additional async thread-safe program compilation, and spares me from
  // having to protect program_map with a lock. And for a project likely to have
  // only a dozen or so cl programs at most should be just fine.
  if (!LoadAndBuildProgram(kCiede2k))
    return false;
  if (!LoadAndBuildProgram(kRGBToLab))
    return false;

  return true;
}

bool CLDeviceContext::LoadAndBuildProgram(Kernels kernel) {
  std::string filename = std::string("cl/") + KernelName(kernel) + ".cl";
  int program_fd = open(filename.c_str(), O_RDONLY);
  if (program_fd < 0)
    return false;

  // figure out size to pre-allocate buffer of correct size
  struct stat program_stat;
  if (fstat(program_fd, &program_stat)) {
    close(program_fd);
    return false;
  }

  std::unique_ptr<char[]> program_bytes(new char[program_stat.st_size + 1]);
  int read_size = read(program_fd, program_bytes.get(), program_stat.st_size);
  close(program_fd);
  if (read_size != program_stat.st_size)
    return false;
  program_bytes[program_stat.st_size] = '\0';

  const char* source_ptr = program_bytes.get();
  int result = 0;
  cl_program program = clCreateProgramWithSource(
      impl_->context, 1, &source_ptr, NULL, &result);
  if (!program || result != CL_SUCCESS)
    return false;

  result = clBuildProgram(program, 0, NULL, "-Werror", NULL, NULL);
  if (result != CL_SUCCESS) {
    std::unique_ptr<char[]> log_char(new char[16384]);
    size_t log_length;
    clGetProgramBuildInfo(program,
                          impl_->device_id,
                          CL_PROGRAM_BUILD_LOG,
                          16384,
                          log_char.get(),
                          &log_length);
    // this should really go in the logs but dump to stdio for now.
    std::cerr << "** failed to build " << KernelName(kernel) << std::endl
              << log_char.get();
    return false;
  }

  return true;
}

const char* CLDeviceContext::KernelName(Kernels kernel) {
  switch (kernel) {
    case kCiede2k:
      return "ciede2k";
    case kRGBToLab:
      return "rgb_to_lab";
    default:
      assert(false);
      return "";
  }
  assert(false);
  return "";
}

std::unique_ptr<CLBuffer> CLDeviceContext::DoMakeBuffer(size_t size) {
  std::unique_ptr<CLBufferImpl> bimpl(new CLBufferImpl);
  if (!bimpl->Setup(size, impl_->context))
    return std::unique_ptr<CLBuffer>();

  return std::unique_ptr<CLBuffer>(bimpl.release());
}

std::unique_ptr<CLCommandQueue> CLDeviceContext::DoMakeCommandQueue() {
  std::unique_ptr<CLCommandQueueImpl> cimpl(new CLCommandQueueImpl);
  if (!cimpl->Setup(impl_->context, impl_->device_id))
    return std::unique_ptr<CLCommandQueue>();

  return std::unique_ptr<CLCommandQueue>(cimpl.release());
}

std::unique_ptr<CLImage> CLDeviceContext::DoMakeImage(const Image* image) {
  std::unique_ptr<CLImageImpl> iimpl(new CLImageImpl(image));
  if (!iimpl->Setup(impl_->context))
    return std::unique_ptr<CLImage>();

  return std::unique_ptr<CLImage>(iimpl.release());
}

std::unique_ptr<CLImage> CLDeviceContext::DoMakeImageFromStrip(
    const PixelStrip* strip) {
  std::unique_ptr<CLImageImpl> iimpl(new CLImageImpl(strip));
  if (!iimpl->Setup(impl_->context))
    return std::unique_ptr<CLImage>();

  return std::unique_ptr<CLImage>(iimpl.release());
}

std::unique_ptr<CLKernel> CLDeviceContext::DoMakeKernel(Kernels kernel) {
  std::unique_ptr<CLKernelImpl> kimpl(new CLKernelImpl);
  if (!kimpl->Setup(impl_->programs[kernel],
                    KernelName(kernel),
                    impl_->context,
                    impl_->device_id))
    return std::unique_ptr<CLKernel>();

  return std::unique_ptr<CLKernel>(kimpl.release());
}

}  // namespace vcsmc
