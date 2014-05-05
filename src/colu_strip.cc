#include "colu_strip.h"

#include <cassert>
#include <cstring>

#include "color.h"

namespace vcsmc {

ColuStrip::ColuStrip() {
  std::memset(colu_, 0x82, sizeof(colu_));
}

ColuStrip::ColuStrip(uint8* colu, uint32 offset) {
  std::memcpy(colu_, colu + offset, kFrameWidthPixels);
}

double ColuStrip::DistanceFrom(ColuStrip* colu_strip) const {
  double accum = 0.0;
  for (uint32 i = 0; i < kFrameWidthPixels; ++i) {
    accum += Color::CartesianDistanceSquaredAtari(
        colu_[i], colu_strip->colu_[i]);
  }
  return accum;
}

void ColuStrip::SetColu(const uint32 pixel, const uint8 colu) {
  assert(pixel < kFrameWidthPixels);
  colu_[pixel] = colu;
}

}  // namespace vcsmc
