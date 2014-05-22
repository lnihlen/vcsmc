#include "pixel_strip.h"

#include <cassert>
#include <cstring>

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "color.h"
#include "colu_strip.h"
#include "pallette.h"

namespace vcsmc {

PixelStrip::PixelStrip(const Image* image, uint32 row_id)
    : width_(image->width()),
      row_id_(row_id),
      pixels_(image->pixels() + (row_id * width_)),
      image_(image) {
}

void PixelStrip::BuildDistances(CLCommandQueue* queue) {
  assert(image_);
  // If we were made from an image, first see if that Image has already been
  // transferred to the GPU.
  CLImage* source_image = image_->cl_image();
  std::unique_ptr<CLImage> strip_image;
  if (!source_image) {
    strip_image = CLDeviceContext::MakeImage(this);
    strip_image->EnqueueCopyToDevice(queue);
    source_image = strip_image.get();
  }
  assert(source_image);

  // Make our output lab buffer
  std::unique_ptr<CLBuffer> lab_strip(
      CLDeviceContext::MakeBuffer(width_ * 4 * sizeof(float)));

  std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(kRGBToTab));
  kernel->SetImageArgument(0, source_image);

  // row is either the row_id from the source image or 0 if we made our own.
  int row = strip_image ? 0 : row_id_;
  kernel->SetByteArgument(1, sizeof(int), &row);
  kernel->SetBufferArgument(2, lab_strip.get());

  kernel->Enqueue(queue);

  error_distances_.reserve(kNTSCColors);
  std::vector<std::unique_ptr<CLBuffer>> error_buffers;
  results_buffers.reserve(kNTSCColors);
  std::vector<std::unique_ptr<CLKernel>> kernels;
  kernels.reserve(kNTSCColors);

  for (size_t i = 0; i < kNTSCColors; ++i) {
    std::unique_ptr<CLBuffer> buffer(CLDeviceContext::MakeBuffer(
        pixel_strip->width() * sizeof(float)));

    std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(kCiede2k));
    kernel->SetBufferArgument(0, pixel_strip->lab_strip());
    kernel->SetBufferArgument(1, Color::AtariLabColorBuffer(i * 2));
    kernel->SetBufferArgument(2, buffer.get());
    kernel->Enqueue(queue);

    std::unique_ptr<float[]> errors(new float[pixel_strip->width()]);
    buffer->EnqueueCopyFromDevice(queue, errors.get());

    error_buffers.push_back(std::move(buffer));
    kernels.push_back(std::move(kernel));
    error_distances_.push_back(std::move(errors));
  }

  // Block until the math is done and our output buffers are valid.
  queue->Finish();
}

void PixelStrip::BuildPallettes(const uint32 max_colus, Random* random) {
  for (uint32 i = 0; i < max_colus; ++i) {
    std::unique_ptr<Pallette> pallete(new Pallete(i));
    pallete->Compute(this, random);
    palletes_.push_back(std::move(pallete));
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
