#ifndef SRC_PALLETTE_H_
#define SRC_PALLETTE_H_

#include "types.h"

namespace vcsmc {

class PixelStrip;

// Associated with a PixelStrip, a Pallette represents a mapping from ABGR color
// values to a fixed number of Atari colors.
class Pallette {
 public:
  // Constructs an emtpy pallette with |num_colus| number of unique colors. Note
  // that brute-force running times increase exponentially with |num_colus|.
  Pallette(uint32 num_colus);

  void Compute(PixelStrip* pixel_strip);

 private:

  const uint32 num_colus_;
};

}  // namespace vcsmc

#endif  // SRC_PALLETTE_H_
