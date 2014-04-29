#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include "types.h"

// Defines a field of uint32 ARGB colors, has a width and height, etc.
class Image {
 public:
  // Constructs an Image that owns a pixel array.
  Image(uint32 width, uint32 height);
  ~Image();

  // 0-based coords.
  void SetPixel(uint32 x, uint32 y, uint32 abgr);
  uint32 GetPixel(uint32 x, uint32 y);

  uint32 width() { return width_; }
  uint32 height() { return height_; }
  const uint32* pixels() { return pixels_; }
  uint32* pixels_writeable() { return pixels_; }

 protected:
  uint32 width_;
  uint32 height_;
  uint32* pixels_;

 private:
  // Private default ctor means "don't call me."
  Image() : width_(0), height_(0), pixels_(nullptr) {}
};

#endif  // SRC_IMAGE_H_
