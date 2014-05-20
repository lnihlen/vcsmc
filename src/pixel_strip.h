#ifndef SRC_PIXEL_STRIP_H_
#define SRC_PIXEL_STRIP_H_

#include <memory>

#include "types.h"

namespace vcsmc {

class CLBuffer;
class CLCommandQueue;
class Image;

class PixelStrip {
 public:
  // Builds an empty PixelStrip of provided width.
  PixelStrip(uint32 width);
  PixelStrip(const Image* image, uint32 row_id);

  bool MakeLabStrip(CLCommandQueue* queue, const Image* image);

  void SetPixel(uint32 pixel, uint32 color);

  const uint32 pixel(uint32 i) const { return pixels_[i]; }
  const uint32 width() const { return width_; }
  const uint32* pixels() const { return pixels_.get(); }
  const uint32 row_id() const { return row_id_; }
  const CLBuffer* lab_strip() const { return lab_strip_.get(); }

 private:
  const uint32 width_;
  const uint32 row_id_;
  std::unique_ptr<uint32[]> pixels_;
  std::unique_ptr<CLBuffer> lab_strip_;
  const Image* image_;
  // If image_ is NULL we make a 1D image to represent this pixelStrip, for
  // conversion to Lab.
  std::unique_ptr<CLImage> strip_image_;
  // Held until the lab_strip_ returns.
  std::unique_ptr<CLKernel> kernel_;
};

}  // namespace vcsmc

#endif  // SRC_PIXEL_STRIP_H_
