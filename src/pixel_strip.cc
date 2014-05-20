#include "pixel_strip.h"

#include <cassert>
#include <cstring>

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "color.h"

namespace vcsmc {

PixelStrip::PixelStrip(const uint32 width, uint32 row_id)
    : width_(width),
      row_id_(row_id),
      pixels_(new uint32[width]),
      image_(NULL) {
  std::memset(pixels_.get(), 0, width * sizeof(uint32));
}

PixelStrip::PixelStrip(const Image* image, uint32 row_id)
    : width_(image->width()),
      row_id_(row_id),
      pixels_(new uint32[image->width()]),
      image_(image) {
  std::memcpy(pixels_.get(),
              image->pixels() + (row_id * width_),
              width * sizeof(uint32));
}

PixelStrip::PixelStrip(const uint32* pixels, const uint32 width, uint32 row_id)
    : width_(width),
      row_id_(row_id),
      pixels_(new uint32[width]) {
  std::memcpy(pixels_.get(), pixels, width * sizeof(uint32));
}

bool PixelStrip::MakeLabStrip(CLCommandQueue* queue) {
  // If we were made from an image, first see if that Image has already been
  // transferred to the GPU.
  CLImage* source_image = image_ ? image_->cl_image() : NULL;
  if (!source_image) {
    strip_image_.reset(CLDeviceContext::MakeImage(this));
    strip_image_->EnqueueCopyToDevice(queue);
    source_image = strip_image.get();
  }
  assert(source_image);

  // Make our output lab buffer
  lab_strip_.reset(CLDeviceContext::MakeBuffer(width_ * 4 * sizeof(float)));
  if (!lab_strip_)
    return false;

  kernel_ = CLDeviceContext::MakeKernel(kRGBToTab);
  if (!kernel_)
    return false;

  if (!kernel_->SetImageArgument(0, source_image))
    return false;

  // row is either the row_id from the source image or 0 if we made our own.
  int row = strip_image_ ? 0 : row_id_;
  if (!kernel_->SetByteArgument(1, sizeof(int), &row))
    return false;

  if (!kernel_->SetBufferArgument(2, lab_strip_.get()))
    return false;

  // Enqueue the conversion kernel.
  return kernel->Enqueue(queue);
}

void PixelStrip::SetPixel(uint32 pixel, uint32 color) {
  assert(pixel < width_);
  pixels_[pixel] = color;
}

}  // namespace vcsmc
