#ifndef SRC_PIXEL_STRIP_H_
#define SRC_PIXEL_STRIP_H_

#include <memory>
#include <vector>

#include "types.h"

namespace vcsmc {

class CLBuffer;
class CLCommandQueue;
class CLImage;
class CLKernel;
class Image;
class Pallette;
class Random;

class PixelStrip {
 public:
  // Builds an empty PixelStrip of provided width.
  PixelStrip(uint32 width);
  PixelStrip(const Image* image, uint32 row_id);

  bool MakeLabStrip(CLCommandQueue* queue, const Image* image);
  void BuildPallettes(CLCommandQueue* queue, uint32 max_colus, Random* random);

  void SetPixel(uint32 pixel, uint32 color);

  const uint32 pixel(uint32 i) const { return pixels_[i]; }
  const uint32 width() const { return width_; }
  const uint32* pixels() const { return pixels_.get(); }
  const uint32 row_id() const { return row_id_; }
  const CLBuffer* lab_strip() const { return lab_strip_.get(); }
  const Pallette* pallette(uint32 i) const { return pallettes_[i].get(); }

 private:
  const uint32 width_;
  const uint32 row_id_;
  std::unique_ptr<uint32[]> pixels_;
  std::unique_ptr<CLBuffer> lab_strip_;
  const Image* image_;
  std::vector<std::unique_ptr<Pallette>> pallettes_;

  // Temporary resources we must retain until we are sure lab_strip_ is
  // ready, or the life of the Strip if easier.
  std::unique_ptr<CLImage> strip_image_;
  std::unique_ptr<CLKernel> kernel_;
};

}  // namespace vcsmc

#endif  // SRC_PIXEL_STRIP_H_
