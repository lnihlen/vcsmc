#ifndef SRC_PALETTE_H_
#define SRC_PALETTE_H_

#include <memory>
#include <vector>

#include "types.h"

namespace vcsmc {

class CLBuffer;
class CLCommandQueue;
class Random;

class Palette {
 public:
  // Constructs an empty palette with |num_colus| number of unique colors. Note
  // that brute-force running times increase exponentially with |num_colus|.
  Palette(uint32 num_colus);

  // Use a clustering algorithm to build a palette with at most |num_colus|
  // colors based on |error_buffer| color distance values, which is assumed to
  // be kFrameWidthPixels x kNTSCColors in dimension so that each row gives
  // per-pixel errors for a color given by the column number. We allow |random|
  // dependency injection to use a fixed seed for the initial classes, if
  // desired.
  void Compute(CLCommandQueue* queue, const CLBuffer* error_buffer,
      Random* random);

  // Atari colu value for the ith class in [0, num_colus). Note that these are
  // sorted by frequency from most frequent class to least.
  const uint8 colu(uint32 i) const { return colus_[i] * 2; }
  // Number of colors in this palette.
  const uint32 num_colus() const { return num_colus_; }
  // Total error in this palette approximation.
  const float error() const { return error_; }
  // The minimum error class for the ith pixel in [0, kFrameWidthPixels)
  const uint8 colu_class(uint32 i) const { return classes_[i]; }

 private:
  // Given a precomputed array of error distances of each pixel to each Atari
  // color and a vector of current colu indices that are the classes, populates
  // |classes_| with the index in |colus_| that is the minimum error class for
  // each pixel. Returns the total error for this classification.
//  float Classify(const PixelStrip* pixel_strip);

  // For each of the classes of colors, find minimum error color to approximate
  // those pixels. Build array of kNTSCColors floats of total error for each K
  // color. For each pixel in strip add error distance to the kth color total
  // error in each color. Then find min error color and use it as new color.
//  void Color(const PixelStrip* pixel_strip);

  const uint32 num_colus_;
  float error_;
  std::vector<uint8> colus_;
  std::vector<uint8> classes_;
};

}  // namespace vcsmc

#endif  // SRC_PALETTE_H_
