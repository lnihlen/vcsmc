#ifndef SRC_PIXEL_STRIP_H_
#define SRC_PIXEL_STRIP_H_

// A PixelStrip defines a read-only row of ABGR colors from an image. It
// supports various techniques for analyzing the colors in the row.

#include <memory>
#include <vector>

#include "types.h"

namespace vcsmc {

class CLBuffer;
class CLCommandQueue;
class CLImage;
class CLKernel;
class ColuStrip;
class Image;
class Pallette;
class Random;

class PixelStrip {
 public:
  PixelStrip(const Image* image, uint32 row_id);

  // Use OpenCL to compute the error distance between every pixel in the strip
  // and every color in the Atari spectrum. Call before calling
  // BuildPallettes().
  void BuildDistances(CLCommandQueue* queue);
  void BuildPallettes(const uint32 max_colus, Random* random);

  float DistanceFrom(ColuStrip* colu_strip);

  const uint32 pixel(uint32 i) const { return pixels_[i]; }
  const uint32 width() const { return width_; }
  const uint32* pixels() const { return pixels_; }
  const uint32 row_id() const { return row_id_; }
  const Pallette* pallette(uint32 i) const { return pallettes_[i - 1].get(); }
  // Returns the error distance for |pixel| from the provided Atari |color|.
  const float distance(uint32 pixel, uint8 color) const {
    return distances_[color / 2][pixel];
  }

 private:
  const uint32 width_;
  const uint32 row_id_;
  const uint32* pixels_;
  const Image* image_;
  // Array of [color / 2][width_] float error distances.
  std::vector<std::unique_ptr<float[]>> distances_;
  std::vector<std::unique_ptr<Pallette>> pallettes_;
};

}  // namespace vcsmc

#endif  // SRC_PIXEL_STRIP_H_
