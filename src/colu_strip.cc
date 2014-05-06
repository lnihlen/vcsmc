#include "colu_strip.h"

#include <cassert>
#include <cstring>

#include "color.h"

namespace vcsmc {

// Blue color is useful for finding unpainted values. Consder using one of the
// odd-numbered colors.
ColuStrip::ColuStrip() {
  std::memset(colu_, 0x82, sizeof(colu_));
}

void ColuStrip::SetColu(const uint32 pixel, const uint8 colu) {
  assert(pixel < kFrameWidthPixels);
  colu_[pixel] = colu;
}

}  // namespace vcsmc
