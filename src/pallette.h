#ifndef SRC_PALLETTE_H_
#define SRC_PALLETTE_H_

#include <memory>
#include <vector>

#include "types.h"

namespace vcsmc {

class CLCommandQueue;
class PixelStrip;
class Random;

// Associated with a PixelStrip, a Pallette represents a mapping from ABGR color
// values to a fixed number of Atari colors.
class Pallette {
 public:
  // Constructs an emtpy pallette with |num_colus| number of unique colors. Note
  // that brute-force running times increase exponentially with |num_colus|.
  Pallette(uint32 num_colus);

  void Compute(const PixelStrip* pixel_strip, Random* random);

  // Atari colu value for the ith class in [0, num_colus).
  const uint8 colu(uint32 i) const { return colus_[i] * 2; }
  // Number of colors in this pallette.
  const uint32 num_colus() const { return num_colus_; }
  // Total error in this pallette approximation.
  const float error() const { return error_; }
  // The minimum error class for the ith pixel in [0, pixel_strip->width())
  const uint8 colu_class(uint32 i) const { return classes_[i]; }

 private:
  // Given a precomputed array of error distances of each pixel to each Atari
  // color and a vector of current colu indicies that are the classes, populates
  // |classes_| with the index in |colus_| that is the minimum error class for
  // each pixel. Returns the total error for this classfication.
  float Classify(const PixelStrip* pixel_strip);

  // For each of the classes of colors, find minimum error color to approximate
  // those pixels. Build array of kNTSCColors floats of total error for each K
  // color. For each pixel in strip add error distance to the kth color total
  // error in each color. Then find min error color and use it as new color.
  void Color(const PixelStrip* pixel_strip);

  const uint32 num_colus_;
  float error_;
  std::vector<uint8> colus_;
  std::vector<uint8> classes_;
};

}  // namespace vcsmc

#endif  // SRC_PALLETTE_H_
