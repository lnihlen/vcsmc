#ifndef SRC_COLOR_H_
#define SRC_COLOR_H_

#include <vector>

#include "types.h"

namespace vcsmc {

class CLBuffer;

// Utilities for converting to/from Atari 2600 Color Codes and ABGR color words
class Color {
 public:
  static const uint32 AtariColorToABGR(uint8 atari_color);
  // Returns a pointer to 4 floats L, a, b, 1.0
  static const float* AtariColorToLab(uint8 atari_color);
};

}  // namespace vcsmc

#endif  // SRC_COLOR_H_
