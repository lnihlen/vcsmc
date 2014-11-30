#include "pixel_strip.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "cl_kernel.h"
#include "cl_program.h"
#include "color.h"
#include "colu_strip.h"
#include "constants.h"
#include "image.h"
#include "palette.h"

namespace vcsmc {

PixelStrip::PixelStrip(const Image* image, uint32 row_id)
    : width_(image->width()),
      row_id_(row_id),
      pixels_(image->pixels() + (row_id * image->width())),
      image_(image) {
}

void PixelStrip::BuildDistances(CLCommandQueue* queue) {
  assert(image_);

  // First we need to convert our image RGB color strip to Lab color.
  const CLImage* source_image = image_->cl_image();
  assert(source_image);

  std::unique_ptr<CLBuffer> lab_strip(
      CLDeviceContext::MakeBuffer(width_ * 4 * sizeof(float)));
  assert(lab_strip);

  std::unique_ptr<CLKernel> kernel(
      CLDeviceContext::MakeKernel(CLProgram::Programs::kRGBToLab));
  assert(kernel);

  // OpenCL images count bottom-up, but we count top-down.
  int inverse_row = image_->height() - row_id_ - 1;
  kernel->SetImageArgument(0, source_image);
  kernel->SetByteArgument(1, sizeof(int), &inverse_row);
  kernel->SetBufferArgument(2, lab_strip.get());

  kernel->Enqueue(queue, width_);

  std::unique_ptr<float[]> lab_strip_values(new float[width_ * 4]);
  lab_strip->EnqueueCopyFromDevice(queue, lab_strip_values.get());

  distances_.reserve(kNTSCColors);
  std::vector<std::unique_ptr<CLBuffer>> buffers;
  buffers.reserve(kNTSCColors * 2);
  std::vector<std::unique_ptr<CLKernel>> kernels;
  kernels.reserve(kNTSCColors);

  for (size_t i = 0; i < kNTSCColors; ++i) {
    std::unique_ptr<CLBuffer> colu_lab_buffer(CLDeviceContext::MakeBuffer(
        width_ * 4 * sizeof(float)));
    colu_lab_buffer->EnqueueFill(
        queue, Color::AtariColorToLab(i * 2), sizeof(float) * 4);

    std::unique_ptr<CLBuffer> out_buffer(CLDeviceContext::MakeBuffer(
        width_ * sizeof(float)));

    std::unique_ptr<CLKernel> ciede_kernel(
        CLDeviceContext::MakeKernel(CLProgram::Programs::kCiede2k));
    ciede_kernel->SetBufferArgument(0, lab_strip.get());
    ciede_kernel->SetBufferArgument(1, colu_lab_buffer.get());
    ciede_kernel->SetBufferArgument(2, out_buffer.get());
    ciede_kernel->Enqueue(queue, width_);

    std::unique_ptr<CLBuffer> error_buffer(CLDeviceContext::MakeBuffer(
        kFrameWidthPixels * sizeof(float)));

    std::unique_ptr<CLKernel> downsample_kernel(
        CLDeviceContext::MakeKernel(CLProgram::Programs::kDownsampleErrors));
    downsample_kernel->SetBufferArgument(0, out_buffer.get());
    int width_int = static_cast<int>(width_);
    downsample_kernel->SetByteArgument(1, sizeof(int), &width_int);
    const int atari_width_int = static_cast<int>(kFrameWidthPixels);
    downsample_kernel->SetByteArgument(2, sizeof(int), &atari_width_int);
    downsample_kernel->SetBufferArgument(3, error_buffer.get());
    downsample_kernel->Enqueue(queue, width_);

    std::unique_ptr<float[]> errors(new float[kFrameWidthPixels]);
    error_buffer->EnqueueCopyFromDevice(queue, errors.get());

    buffers.push_back(std::move(colu_lab_buffer));
    buffers.push_back(std::move(out_buffer));
    buffers.push_back(std::move(error_buffer));
    kernels.push_back(std::move(ciede_kernel));
    kernels.push_back(std::move(downsample_kernel));
    distances_.push_back(std::move(errors));
  }

  // Block until the math is done and our output buffers are valid.
  queue->Finish();
}

std::unique_ptr<Palette> PixelStrip::BuildPalette(const uint32 max_colus,
      Random* random) {
  std::unique_ptr<Palette> palette(new Palette(max_colus));
  palette->Compute(this, random);
  return std::move(palette);
}

float PixelStrip::DistanceFrom(ColuStrip* colu_strip) {
  float distance = 0;
  for (uint32 i = 0; i < kFrameWidthPixels; ++i)
    distance += Distance(i, colu_strip->colu(i));
  return distance;
}

float PixelStrip::Distance(uint32 pixel, uint8 color) const {
  assert(pixel < kFrameWidthPixels);
  return distances_[color / 2][pixel];
}

}  // namespace vcsmc
