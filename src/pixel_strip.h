#ifndef SRC_PIXEL_STRIP_H_
#define SRC_PIXEL_STRIP_H_

// A PixelStrip defines a read-only row of ABGR colors from an image. It
// supports various techniques for analyzing the colors in the row.

#include <memory>
#include <vector>

#include "types.h"

#error deprecate me

namespace vcsmc {

class CLBuffer;
class CLCommandQueue;
class CLImage;
class CLKernel;
class ColuStrip;
class Image;
class Palette;
class Random;

class PixelStrip {
 public:
  PixelStrip(const Image* image, uint32 row_id);

  // Use OpenCL to compute the error distance between every pixel in the strip
  // and every color in the Atari spectrum. Call before calling
  // BuildPalette().
  void BuildDistances(CLCommandQueue* queue);

  // Use a clustering algorithm 
  std::unique_ptr<Palette> BuildPalette(const uint32 max_colus, Random* random);

  // Returns total error distance from a provided ColuStrip.
  float DistanceFrom(ColuStrip* colu_strip);

  // Returns the error distance for |pixel| from the provided Atari |color|.
  // |pixel| is within [0, kFrameWidthPixels].
  float Distance(uint32 pixel, uint8 color) const;

  uint32 pixel(uint32 i) const { return pixels_[i]; }
  uint32 width() const { return width_; }
  const uint32* pixels() const { return pixels_; }
  uint32 row_id() const { return row_id_; }

 private:
  const uint32 width_;
  const uint32 row_id_;
  const uint32* pixels_;
  const Image* image_;
  // Array of [color / 2][kFrameWidthPixels] float error distances.
  std::vector<std::unique_ptr<float[]>> distances_;
};

}  // namespace vcsmc

#endif  // SRC_PIXEL_STRIP_H_
