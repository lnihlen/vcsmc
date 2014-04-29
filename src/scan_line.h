#ifndef SRC_SCAN_LINE_H_
#define SRC_SCAN_LINE_H_

#include <vector>

#include "player.h"
#include "types.h"

// A ScanLine represents the output of the first pass of the compiler, fitting.
// The goal of fitting is to approximate a single line of the input image with
// the drawing primitives available on the VCS.
class ScanLine {
 public:

 private:
  // 80-bit playfield, 4 pixels/bit.
  uint8 playfield_[10];

  // Background color/luminance.
  uint8 colubk_;

  // Playfield color/luminance.
  uint8 colupf_;

  std::vector<Player> players_;
};

#endif  // SRC_SCAN_LINE_H_
