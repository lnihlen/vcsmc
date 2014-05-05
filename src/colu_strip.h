#ifndef SRC_COLU_STRIP_H_
#define SRC_COLU_STRIP_H_

#include <memory>

#include "constants.h"
#include "types.h"

namespace vcsmc {

// A ColuStrip represents a single line of data in Atari Colors. It can be
// compared to another one for an error distance. Both Frames and ScanLines
// produce them. They can also be appended directly to Images, for logging
// output.
class ColuStrip {
 public:
  ColuStrip();
  // Builds a ColuStrip by making a copy of kFrameWidthPixels bytes from the
  // provided pointer and offset.
  ColuStrip(uint8* colu, uint32 offset);

  // Returns the error distance from the provided ColuStrip.
  double DistanceFrom(ColuStrip* colu_strip) const;

  void SetColu(const uint32 pixel, const uint8 colu);

  // Given a pixel [0..width()) returns the stored colu value.
  const uint8 colu(const uint32 pixel) const { return colu_[pixel]; }
  const uint32 width() const { return kFrameWidthPixels; }

  const uint8* colus() const { return colu_; }

 private:
  uint8 colu_[kFrameWidthPixels];
};

}  // namespace vcsmc

#endif  // SRC_COLU_STRIP_H_
