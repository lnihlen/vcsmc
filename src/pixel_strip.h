#ifndef SRC_PIXEL_STRIP_H_
#define SRC_PIXEL_STRIP_H_

#include <memory>

#include "types.h"

namespace vcsmc {

class PixelStrip {
 public:
  // Builds an empty PixelStrip of provided width.
  PixelStrip(uint32 width);

  // Builds a PixelStrip by making a copy of |width| uint32s from the provided
  // pointer.
  PixelStrip(const uint32* pixels, uint32 width);

  void SetPixel(uint32 pixel, uint32 color);

  // Returns error distance between two pixel strips of equal width.
  double DistanceFrom(const PixelStrip* strip) const;

  const uint32 pixel(uint32 i) const { return pixels_[i]; }
  const uint32 width() const { return width_; }
  const uint32* pixels() const { return pixels_.get(); }

 private:
  const uint32 width_;
  std::unique_ptr<uint32[]> pixels_;
};

}  // namespace vcsmc

#endif  // SRC_PIXEL_STRIP_H_
