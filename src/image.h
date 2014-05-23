#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include <memory>

#include "types.h"

namespace vcsmc {

class CLCommandQueue;
class CLImage;
class ColuStrip;
class PixelStrip;

// Defines a field of uint32 ABGR colors, has a width and height, etc.
class Image {
 public:
  // Constructs an Image that owns a pixel array.
  Image(uint32 width, uint32 height);
  bool CopyToDevice(CLCommandQueue* queue);

  // Builds a copy of our |row| of pixels and returns.
  std::unique_ptr<PixelStrip> GetPixelStrip(uint32 row);
  // Copies contents of |strip| into our own buffer at row.
  void SetStrip(uint32 row, ColuStrip* strip);

  const uint32 width() const { return width_; }
  const uint32 height() const { return height_; }
  const uint32* pixels() const { return pixels_.get(); }
  uint32* pixels_writeable() { return pixels_.get(); }
  const uint32 pixel(uint32 x, uint32 y) const {
    return pixels_[(y * width_) + x];
  }
  const CLImage* cl_image() const { return cl_image_.get(); }

 protected:
  const uint32 width_;
  const uint32 height_;
  std::unique_ptr<uint32[]> pixels_;
  std::unique_ptr<CLImage> cl_image_;

 private:
  // Private default ctor means "don't call me."
  Image() : width_(0), height_(0), pixels_(nullptr) {}
};

}  // namespace vcsmc

#endif  // SRC_IMAGE_H_
