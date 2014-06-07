#ifndef SRC_COLU_STRIP_H_
#define SRC_COLU_STRIP_H_

// A ColuStrip is a row of Atari Color/Luminance values. Produced by States
// as they simulate the TIA. Possibly maintains an error strip, and can
// readily compute the distance per-pixel or total from a PixelStrip.

#include <memory>

#include "constants.h"
#include "range.h"
#include "types.h"

namespace vcsmc {

class ColuStrip {
 public:
  ColuStrip(uint32 row);

  const uint32 colu(uint32 column) const { return colus_[column]; }
  void set_colu(uint32 column, uint8 colu) { colus_[column] = colu; }
  const Range& range() const { return range_; }

 private:
  Range range_;
  uint8 colus_[kFrameWidthPixels];
};

}  // namespace vcsmc

#endif
