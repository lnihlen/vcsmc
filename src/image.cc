#include "image.h"

#include <cassert>

#include "pixel_strip.h"

namespace vcsmc {

Image::Image(uint32 width, uint32 height)
    : width_(width),
      height_(height),
      pixels_(new uint32[width * height]) {
}

void Image::SetPixel(uint32 x, uint32 y, uint32 abgr) {
  *(pixels_.get() + ((y * width_) + x)) = abgr;
}

std::unique_ptr<PixelStrip> Image::GetPixelStrip(uint32 row) {
  assert(row < height_);
  return std::unique_ptr<PixelStrip>(
      new PixelStrip(pixels_.get() + (row * width_), width_));
}

}  // namespace vcsmc
