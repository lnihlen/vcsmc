#ifndef SRC_PIXEL_STRIP_H_
#define SRC_PIXEL_STRIP_H_

#include <memory>

#include "types.h"

namespace vcsmc {

// Like ColuStrip but for images. Represents a single row of pixels from an
// Image. Has an arbitrary width.

class PixelStrip {
 public:
  // Builds a PixelStrip by making a copy of |width| uint32s from the provided
  // pointer.
  PixelStrip(const uint32* pixels, uint32 width);

  const uint32 pixel(uint32 i) const { return pixels_[i]; }
  const uint32 width() const { return width_; }

 private:
  const uint32 width_;
  std::unique_ptr<uint32[]> pixels_;
};

}  // namespace vcsmc

#endif  // SRC_PIXEL_STRIP_H_
