#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include <memory>

#include "types.h"

namespace vcsmc {

// Defines a field of uint32 ABGR colors, has a width and height, etc. On
// little-endian machines the bytes will be RGBA in order from lowest address
// to highest, meaning that the A will end up in the most significant byte of
// the uint32.
class Image {
 public:
  // Constructs an Image that owns a pixel array.
  Image(uint32 width, uint32 height);
  ~Image();

  uint32 width() const { return width_; }
  uint32 height() const { return height_; }
  uint32* pixels() const { return pixels_.get(); }
  uint32* pixels_writeable() { return pixels_.get(); }
  uint32 pixel(uint32 x, uint32 y) const {
    return pixels_[(y * width_) + x];
  }

 private:
  const uint32 width_;
  const uint32 height_;
  std::unique_ptr<uint32[]> pixels_;

  // Private default ctor means "don't call me."
  Image() : width_(0), height_(0), pixels_(nullptr) {}
};

}  // namespace vcsmc

#endif  // SRC_IMAGE_H_
