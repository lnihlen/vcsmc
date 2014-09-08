#include "pixel_strip.h"

#include <cassert>
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
#include "pallette.h"

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

  kernel->Enqueue(queue);

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

    std::unique_ptr<CLKernel> kernel(
        CLDeviceContext::MakeKernel(CLProgram::Programs::kCiede2k));
    kernel->SetBufferArgument(0, lab_strip.get());
    kernel->SetBufferArgument(1, colu_lab_buffer.get());
    kernel->SetBufferArgument(2, out_buffer.get());
    kernel->Enqueue(queue);

    std::unique_ptr<float[]> errors(new float[width_]);
    out_buffer->EnqueueCopyFromDevice(queue, errors.get());

    buffers.push_back(std::move(colu_lab_buffer));
    buffers.push_back(std::move(out_buffer));
    kernels.push_back(std::move(kernel));
    distances_.push_back(std::move(errors));
  }

  // Block until the math is done and our output buffers are valid.
  queue->Finish();
}

void PixelStrip::BuildPallettes(const uint32 max_colus, Random* random) {
  for (uint32 i = 1; i <= max_colus; ++i) {
    std::unique_ptr<Pallette> pallette(new Pallette(i));
    pallette->Compute(this, random);
    pallettes_.push_back(std::move(pallette));
  }
}

float PixelStrip::DistanceFrom(ColuStrip* colu_strip) {
  assert(width_ / 2 == kFrameWidthPixels);
  float distance = 0;
  for (uint32 i = 0; i < width_; ++i)
    distance += distances_[colu_strip->colu(i / 2) / 2][i];
  return distance;
}

}  // namespace vcsmc
