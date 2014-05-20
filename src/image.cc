#include "image.h"

#include <cassert>
#include <cstring>

#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "pixel_strip.h"

namespace vcsmc {

Image::Image(uint32 width, uint32 height)
    : width_(width),
      height_(height),
      pixels_(new uint32[width * height]) {
}

bool Image::CopyToDevice(CLCommandQueue* queue) {
  cl_image_.reset(CLDeviceContext::MakeImage(this));
  if (!cl_image_)
    return false;
  return cl_image_->EnqueueCopyToDevice(queue);
}

void Image::SetPixel(uint32 x, uint32 y, uint32 abgr) {
  *(pixels_.get() + ((y * width_) + x)) = abgr;
}

std::unique_ptr<PixelStrip> Image::GetPixelStrip(uint32 row) {
  assert(row < height_);
  return std::unique_ptr<PixelStrip>(
    new PixelStrip(this, row));
}

void Image::SetStrip(uint32 row, PixelStrip* strip) {
  assert(row < height_);
  assert(strip->width() == width_);
  std::memcpy(pixels_.get() + (row * width_),
              strip->pixels(),
              width_ * sizeof(uint32));
}

}  // namespace vcsmc
