#ifndef SRC_COLOR_H_
#define SRC_COLOR_H_

#include "types.h"

namespace vcsmc {

// Utilities for converting to/from Atari 2600 Color Codes and ABGR color words
class Color {
 public:
  static uint32 AtariColorToABGR(uint8 atari_color);
  static uint8 ABGRToAtariColor(uint32 abgr);
  // Difference between two colors as calculated by Cartesian distance in ABGR
  // space. Alpha is ignored.
  static double CartesianDistanceSquaredABGR(uint32 a, uint32 b);
  static double CartesianDistanceSquaredAtari(uint8 a, uint8 b);

 private:
};

}  // namespace vcsmc

#endif  // SRC_COLOR_H_
