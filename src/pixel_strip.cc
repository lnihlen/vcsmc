#include "pixel_strip.h"

#include <cassert>
#include <cstring>

#include "color.h"
#include "histogram.h"

namespace vcsmc {

PixelStrip::PixelStrip(const uint32 width)
    : width_(width),
      pixels_(new uint32[width]),
      histo_(new Histogram) {
  std::memset(pixels_.get(), 0, width * sizeof(uint32));
}

PixelStrip::PixelStrip(const uint32* pixels, const uint32 width)
    : width_(width),
      pixels_(new uint32[width]),
      histo_(new Histogram) {
  std::memcpy(pixels_.get(), pixels, width * sizeof(uint32));
}

void PixelStrip::SetPixel(uint32 pixel, uint32 color) {
  assert(pixel < width_);
  pixels_[pixel] = color;
}

double PixelStrip::DistanceFrom(const PixelStrip* strip) const {
  assert(width_ == strip->width_);
  double accum = 0.0;
  for (uint32 i = 0; i < width_; ++i) {
    accum += Color::CartesianDistanceSquaredABGR(pixels_[i], strip->pixels_[i]);
  }
  return accum;
}

void PixelStrip::BuildHistogram() {
  assert(histo_);
  histo_->Compute(this);
}

}  // namespace vcsmc
