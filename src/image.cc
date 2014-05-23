#include "image.h"

#include <cassert>
#include <cstring>

#include "cl_command_queue.h"
#include "cl_buffer.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "cl_kernel.h"
#include "color.h"
#include "colu_strip.h"
#include "constants.h"
#include "pallette.h"
#include "pixel_strip.h"

namespace vcsmc {

Image::Image(uint32 width, uint32 height)
    : width_(width),
      height_(height),
      pixels_(new uint32[width * height]) {
}

bool Image::CopyToDevice(CLCommandQueue* queue) {
  cl_image_ = CLDeviceContext::MakeImage(this);
  if (!cl_image_)
    return false;
  return cl_image_->EnqueueCopyToDevice(queue);
}

std::unique_ptr<PixelStrip> Image::GetPixelStrip(uint32 row) {
  assert(row < height_);
  return std::unique_ptr<PixelStrip>(
    new PixelStrip(this, row));
}

void Image::SetStrip(uint32 row, ColuStrip* strip) {
  assert(row < height_);
  assert(width_ / 2 == kFrameWidthPixels);

  uint32* px = pixels_.get() + (row * width_);
  for (uint32 i = 0; i < kFrameWidthPixels; ++i) {
    uint32 colu_abgr = Color::AtariColorToABGR(strip->colu(i));
    *(px++) = colu_abgr;
    *(px++) = colu_abgr;
  }
}

}  // namespace vcsmc
