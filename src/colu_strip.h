#ifndef SRC_COLU_STRIP_H_
#define SRC_COLU_STRIP_H_

#include "constants.h"
#include "types.h"

namespace vcsmc {

// A ColuStrip represents a single line of data in Atari Colors. It can be
// compared to another one for an error distance. Both Frames and ScanLines
// produce them. They can also be appended directly to Images, for logging
// output.
class ColuStrip {
 public:
  // Given a pixel [0..width()) returns the stored colu value.
  const uint8 colu(const uint32 pixel) const { return colu_[pixel]; }
  const uint32 width() const { return kFrameWidthPixels; }

 private:
  uint8 colu_[kFrameWidthPixels];
};

}  // namespace vcsmc

#endif  // SRC_COLU_STRIP_H_
