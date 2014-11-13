// smap - saliency map generator

#include <list>
#include <string>

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_kernel.h"
#include "cl_program.h"
#include "gray_map.h"
#include "types.h"

// With thanks to Sean Aaron Anderson
// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
uint32 NextHighestPower(uint32 v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    // optional keyframe ffmpeg csv file too!
    return -1;
  }

  std::string input_file_path(argv[1]);
  std::string output_file_path(argv[2]);

  if (!vcsmc::CLDeviceContext::Setup()) {
    printf("OpenCL setup failed!\n");
    return -1;
  }

  std::unique_ptr<vcsmc::GrayMap> input_map = vcsmc::GrayMap::Load(
      input_file_path);
  if (!input_map) {
    printf("Error opening %s\n", input_file_path.c_str());
    return -1;
  }

  uint32 width = input_map->width();
  uint32 height = input_map->height();
  uint32 power_width = NextHighestPower(input_map->width());
  uint32 power_height = NextHighestPower(input_map->height());

  // Make containers for OpenCL outputs.
  float mean = -1.0f;
  std::unique_ptr<vcsmc::GrayMap> output_map(
      new vcsmc::GrayMap(input_map->width(), input_map->height()));

  // Nested scope so that OpenCL wrapper objects get destructed before OpenCL
  // Teardown call at bottom of function.
  {
    std::unique_ptr<vcsmc::CLCommandQueue> queue =
        vcsmc::CLDeviceContext::MakeCommandQueue();
    std::unique_ptr<vcsmc::CLBuffer> first_buffer(
        vcsmc::CLDeviceContext::MakeBuffer(
            power_width * power_height * 2 * sizeof(float)));

    std::unique_ptr<vcsmc::CLBuffer> map_buffer(
        vcsmc::CLDeviceContext::MakeBuffer(width * height * sizeof(float)));
    map_buffer->EnqueueCopyToDevice(queue.get(), input_map->values());
    std::unique_ptr<vcsmc::CLKernel> unpack_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kUnpackRealToComplex));
    unpack_kernel->SetBufferArgument(0, map_buffer.get());
    unpack_kernel->SetByteArgument(1, sizeof(uint32), &width);
    unpack_kernel->SetByteArgument(2, sizeof(uint32), &height);
    unpack_kernel->SetByteArgument(3, sizeof(uint32), &power_width);
    unpack_kernel->SetByteArgument(4, sizeof(uint32), &power_height);
    unpack_kernel->SetBufferArgument(5, first_buffer.get());
    unpack_kernel->Enqueue(queue.get(), unpack_kernel->WorkGroupSize());

    std::unique_ptr<vcsmc::CLBuffer> second_buffer(
        vcsmc::CLDeviceContext::MakeBuffer(
            power_width * power_height * 2 * sizeof(float)));
    std::list<std::unique_ptr<vcsmc::CLKernel>> kernels;

    // Forward row FFT
    bool up = true;
    uint32 p = 1;
    while (p < power_width) {
      std::unique_ptr<vcsmc::CLKernel> kernel(
          vcsmc::CLDeviceContext::MakeKernel(
              vcsmc::CLProgram::Programs::kFFTRadix2));
      kernel->SetBufferArgument(0, up ?
          first_buffer.get() : second_buffer.get());
      kernel->SetByteArgument(1, sizeof(uint32), &p);
      uint32 output_stride = (p == power_width / 2) ? power_height : 1;
      kernel->SetByteArgument(2, sizeof(uint32), &output_stride);
      kernel->SetBufferArgument(3, up ?
          second_buffer.get() : first_buffer.get());
      kernel->Enqueue2D(queue.get(), power_width / 2, power_height);
      kernels.push_back(std::move(kernel));
      up = !up;
      p = p << 1;
    }

    // Forward column FFT
    p = 1;
    while (p < power_height) {
      std::unique_ptr<vcsmc::CLKernel> kernel(
          vcsmc::CLDeviceContext::MakeKernel(
              vcsmc::CLProgram::Programs::kFFTRadix2));
      kernel->SetBufferArgument(0, up ?
          first_buffer.get() : second_buffer.get());
      kernel->SetByteArgument(1, sizeof(uint32), &p);
      uint32 output_stride = (p == power_height / 2) ? power_width : 1;
      kernel->SetByteArgument(2, sizeof(uint32), &output_stride);
      kernel->SetBufferArgument(3, up ?
          second_buffer.get() : first_buffer.get());
      kernel->Enqueue2D(queue.get(), power_height / 2, power_width);
      kernels.push_back(std::move(kernel));
      up = !up;
      p = p << 1;
    }

    // Spectral Residual
    std::unique_ptr<vcsmc::CLKernel> sr_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kSpectralResidual));
    sr_kernel->SetBufferArgument(0, up ?
        first_buffer.get() : second_buffer.get());
    sr_kernel->SetBufferArgument(1, up ?
        second_buffer.get() : first_buffer.get());
    sr_kernel->Enqueue2D(queue.get(), power_width, power_height);
    up = !up;

    // Inverse row FFT
    std::unique_ptr<vcsmc::CLKernel> ir_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kInverseFFTNormalize));
    ir_kernel->SetBufferArgument(0, up ?
        first_buffer.get() : second_buffer.get());
    float norm = 1.0f;
    ir_kernel->SetByteArgument(1, sizeof(float), &norm);
    ir_kernel->SetBufferArgument(2, up ?
        second_buffer.get() : first_buffer.get());
    ir_kernel->Enqueue2D(queue.get(), power_width / 2, power_height);
    up = !up;
    p = 1;
    while (p < power_width) {
      std::unique_ptr<vcsmc::CLKernel> kernel(
          vcsmc::CLDeviceContext::MakeKernel(
              vcsmc::CLProgram::Programs::kFFTRadix2));
      kernel->SetBufferArgument(0, up ?
          first_buffer.get() : second_buffer.get());
      kernel->SetByteArgument(1, sizeof(uint32), &p);
      uint32 output_stride = (p == power_width / 2) ? power_height : 1;
      kernel->SetByteArgument(2, sizeof(uint32), &output_stride);
      kernel->SetBufferArgument(3, up ?
          second_buffer.get() : first_buffer.get());
      kernel->Enqueue2D(queue.get(), power_width / 2, power_height);
      kernels.push_back(std::move(kernel));
      up = !up;
      p = p << 1;
    }
    // The data are transposed so we deal with it in transposed rows but still
    // need to normalize with N = width, not height.
    std::unique_ptr<vcsmc::CLKernel> irt_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kInverseFFTNormalize));
    irt_kernel->SetBufferArgument(0, up ?
        first_buffer.get() : second_buffer.get());
    norm = (float)power_width;
    irt_kernel->SetByteArgument(1, sizeof(float), &norm);
    irt_kernel->SetBufferArgument(2, up ?
        second_buffer.get() : first_buffer.get());
    irt_kernel->Enqueue2D(queue.get(), power_height / 2, power_width);
    up = !up;

    // Inverse column FFT
    std::unique_ptr<vcsmc::CLKernel> ic_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kInverseFFTNormalize));
    ic_kernel->SetBufferArgument(0, up ?
        first_buffer.get() : second_buffer.get());
    norm = 1.0f;
    ic_kernel->SetByteArgument(1, sizeof(float), &norm);
    ic_kernel->SetBufferArgument(2, up ?
        second_buffer.get() : first_buffer.get());
    ic_kernel->Enqueue2D(queue.get(), power_height / 2, power_width);
    up = !up;
    p = 1;
    while (p < power_height) {
      std::unique_ptr<vcsmc::CLKernel> kernel(
          vcsmc::CLDeviceContext::MakeKernel(
              vcsmc::CLProgram::Programs::kFFTRadix2));
      kernel->SetBufferArgument(0, up ?
          first_buffer.get() : second_buffer.get());
      kernel->SetByteArgument(1, sizeof(uint32), &p);
      uint32 output_stride = (p == power_height / 2) ? power_width : 1;
      kernel->SetByteArgument(2, sizeof(uint32), &output_stride);
      kernel->SetBufferArgument(3, up ?
          second_buffer.get() : first_buffer.get());
      kernel->Enqueue2D(queue.get(), power_height / 2, power_width);
      kernels.push_back(std::move(kernel));
      up = !up;
      p = p << 1;
    }
    std::unique_ptr<vcsmc::CLKernel> kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kInverseFFTNormalize));
    kernel->SetBufferArgument(0, up ?
        first_buffer.get() : second_buffer.get());
    norm = (float)power_height;
    kernel->SetByteArgument(1, sizeof(float), &norm);
    kernel->SetBufferArgument(2, up ?
        second_buffer.get() : first_buffer.get());
    kernel->Enqueue2D(queue.get(), power_width / 2, power_height);
    up = !up;

    // Drop imaginary values and zero padding.
    std::unique_ptr<vcsmc::CLKernel> pack_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kPackComplexToReal));
    pack_kernel->SetBufferArgument(0,
        up ? first_buffer.get() : second_buffer.get());
    pack_kernel->SetByteArgument(1, sizeof(uint32), &power_width);
    pack_kernel->SetByteArgument(2, sizeof(uint32), &power_height);
    pack_kernel->SetByteArgument(3, sizeof(uint32), &width);
    pack_kernel->SetByteArgument(4, sizeof(uint32), &height);
    pack_kernel->SetBufferArgument(5, map_buffer.get());
    pack_kernel->Enqueue(queue.get(), pack_kernel->WorkGroupSize());

    // Square values to get spectral residual.
    std::unique_ptr<vcsmc::CLBuffer> square_buffer(
        vcsmc::CLDeviceContext::MakeBuffer(width * height * sizeof(float)));
    std::unique_ptr<vcsmc::CLKernel> square_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kSquare));
    square_kernel->SetBufferArgument(0, map_buffer.get());
    square_kernel->SetBufferArgument(1, square_buffer.get());
    square_kernel->Enqueue(queue.get(), width * height);

    // Copy back out to spectral residual picture buffer.
    square_buffer->EnqueueCopyFromDevice(queue.get(),
       output_map->values_writeable());

    // Sum squared error to get mean.
    std::unique_ptr<vcsmc::CLKernel> sum_kernel(
        vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::Programs::kSum));
    std::unique_ptr<vcsmc::CLBuffer> sum_buffer(
        vcsmc::CLDeviceContext::MakeBuffer(sizeof(float)));
    sum_kernel->SetBufferArgument(0, square_buffer.get());
    uint32 sum_length = width * height;
    sum_kernel->SetByteArgument(1, sizeof(uint32), &sum_length);
    sum_kernel->SetByteArgument(2,
        sizeof(float) * sum_kernel->WorkGroupSize(), NULL);
    sum_kernel->SetBufferArgument(3, sum_buffer.get());
    sum_kernel->Enqueue(queue.get(), sum_kernel->WorkGroupSize());

    sum_buffer->EnqueueCopyFromDevice(queue.get(), &mean);
    queue->Finish();

    mean = mean / (float)sum_length;
  }

  output_map->Save(output_file_path);
  printf("mean: %f\n", mean);

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
