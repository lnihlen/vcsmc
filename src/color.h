#ifndef SRC_COLOR_H_
#define SRC_COLOR_H_

#include "types.h"

// Utilities for converting to/from Atari 2600 Color Codes and ABGR color words
class Color {
 public:
  static uint32 AtariColorToABGR(uint8 atari_color);
  static uint8 ABGRToAtariColor(uint32 abgr);

 private:
  // Difference between two colors as calculated by Cartesian distance in ABGR
  // space. Alpha is ignored.
  static double CartesianDistanceSquaredABGR(uint32 a, uint32 b);
};

#endif  // SRC_COLOR_H_
