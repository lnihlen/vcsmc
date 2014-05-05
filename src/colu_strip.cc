#include "colu_strip.h"

#include <cstring>

#include "color.h"

namespace vcsmc {

ColuStrip::ColuStrip() {
  std::memset(colu_, 0, sizeof(colu_));
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

}  // namespace vcsmc
