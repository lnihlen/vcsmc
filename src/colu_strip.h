#ifndef SRC_COLU_STRIP_H_
#define SRC_COLU_STRIP_H_

// A ColuStrip is a row of Atari Colur/Luminance values. Produced by ScanLines
// as they simulate the TIA state. Possibly maintains an error strip, and can
// readily compute the distance per-pixel or total from a PixelStrip.

#include <memory>

#include "types.h"

namespace vcsmc {

class ColuStrip {
 public:
  ColuStrip();

  uint colu(uint32 column) { return colus_[column]; }
  void set_colu(uint32 column, uint8 colu) { colus_[column] = colu; }

 private:
  std::unique_ptr<uint8[]> colus_;
};

}  // namespace vcsmc

#endif
