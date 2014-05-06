#include "pixel_strip.h"

#include <cstring>

namespace vcsmc {

PixelStrip::PixelStrip(const uint32* pixels, const uint32 width)
    : width_(width),
      pixels_(new uint32[width]) {
  std::memcpy(pixels_.get(), pixels, width * sizeof(uint32));
}

}  // namespace vcsmc
