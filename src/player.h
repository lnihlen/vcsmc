#ifndef SRC_PLAYER_H_
#define SRC_PLAYER_H_

#include "types.h"

// ScanLine keeps many Player objects, representing a single instance of one
// of the two player graphics primitives on the VCS.
class Player {
 public:

 private:
  // When to trigger this Player bitfield.
  uint32 color_clock_;

  // What color to render this Player in.
  uint8 colup_;

  // Which pixels to paint with with colup_.
  uint8 bitmask_;

  // If this bitmask should repeat, or be scaled.
  uint8 nusiz_;
};

#endif  // SRC_PLAYER_H_
