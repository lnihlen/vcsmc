#include "cl_device_context.h"

#include <cassert>
#include <iostream>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cl_buffer_impl.h"
#include "cl_command_queue_impl.h"
#include "cl_image_impl.h"
#include "cl_include.h"
#include "cl_kernel_impl.h"
#include "cl_program.h"

namespace vcsmc {

CLDeviceContext* CLDeviceContext::instance_ = NULL;

struct CLDeviceContext::Impl {
  cl_device_id device_id;
  cl_context context;
  uint64 local_memory_size;

  // Our cache of previously loaded and compiled programs.
  cl_program programs[CLProgram::Programs::PROGRAM_COUNT];
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
std::unique_ptr<CLImage> CLDeviceContext::MakeImage(uint32 width,
    uint32 height) {
  return instance_->DoMakeImage(width, height);
}

// static
std::unique_ptr<CLKernel> CLDeviceContext::MakeKernel(
    CLProgram::Programs program) {
  return instance_->DoMakeKernel(program);
}

// static
uint64 CLDeviceContext::LocalMemorySize() {
  return instance_->impl_->local_memory_size;
}

CLDeviceContext::CLDeviceContext() : impl_(new Impl) {
}

CLDeviceContext::~CLDeviceContext() {
  for (size_t i = 0; i < CLProgram::Programs::PROGRAM_COUNT; ++i) {
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

  for (uint32 i = 0; i < CLProgram::Programs::PROGRAM_COUNT; ++i) {
    if (!BuildProgram(static_cast<CLProgram::Programs>(i)))
      return false;
  }

  // Query for local memory size.
  result = clGetDeviceInfo(impl_->device_id, CL_DEVICE_LOCAL_MEM_SIZE,
      sizeof(uint64), &impl_->local_memory_size, NULL);

  return result == CL_SUCCESS;
}

bool CLDeviceContext::BuildProgram(CLProgram::Programs program) {
  std::string source = CLProgram::GetProgramString(program);
  const char* source_cstr = source.c_str();
  int result = 0;
  cl_program prog = clCreateProgramWithSource(
      impl_->context, 1, &source_cstr, NULL, &result);
  if (!prog || result != CL_SUCCESS)
    return false;

  result = clBuildProgram(prog, 0, NULL, "-Werror", NULL, NULL);
  if (result != CL_SUCCESS) {
    std::unique_ptr<char[]> log_char(new char[16384]);
    size_t log_length;
    clGetProgramBuildInfo(prog,
                          impl_->device_id,
                          CL_PROGRAM_BUILD_LOG,
                          16384,
                          log_char.get(),
                          &log_length);
    // this should really go in the logs but dump to stdio for now.
    std::cerr << "** failed to build " << CLProgram::GetProgramName(program)
              << std::endl << log_char.get();
    return false;
  }

  impl_->programs[program] = prog;
  return true;
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

std::unique_ptr<CLImage> CLDeviceContext::DoMakeImage(
    uint32 width, uint32 height) {
  std::unique_ptr<CLImageImpl> image_impl(new CLImageImpl(width, height));
  if (!image_impl->Setup(impl_->context))
    return std::unique_ptr<CLImage>();

  return std::unique_ptr<CLImage>(image_impl.release());
}

std::unique_ptr<CLKernel> CLDeviceContext::DoMakeKernel(
    CLProgram::Programs program) {
  std::unique_ptr<CLKernelImpl> kimpl(new CLKernelImpl);
  if (!kimpl->Setup(impl_->programs[program],
                    CLProgram::GetProgramName(program).c_str(),
                    impl_->context,
                    impl_->device_id))
    return std::unique_ptr<CLKernel>();

  return std::unique_ptr<CLKernel>(kimpl.release());
}

}  // namespace vcsmc
