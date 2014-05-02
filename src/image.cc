#include "image.h"

namespace vcsmc {

Image::Image(uint32 width, uint32 height)
    : width_(width),
      height_(height),
      pixels_(new uint32[width * height]) {
}

void Image::SetPixel(uint32 x, uint32 y, uint32 abgr) {
  *(pixels_.get() + ((y * width_) + x)) = abgr;
}

uint32 Image::GetPixel(uint32 x, uint32 y) {
  return pixels_[(y * width_) + x];
}

}  // namespace vcsmc
