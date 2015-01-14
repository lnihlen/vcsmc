#ifndef SRC_RANDOM_H_
#define SRC_RANDOM_H_

#include "types.h"

// Uses public-domain implementation provided by Chris Lomont www.lomont.org

namespace vcsmc {

class Random {
 public:
  // seed points to 16 uint32 with seed state.
  Random(const uint32* seed);
  // Reads 16 bytes from /dev/urandom.
  Random();

  // Returns the next random number in the sequence.
  uint32 Next();

  // Returns floating point number uniformly distributed between [0..1].
  float NextFloat();

  // Returns pointer to 16 uint32s
  const uint32* state() const { return state_; }

 private:
  uint32 state_[16];
  uint32 index_;
};

}  // namespace vcsmc

#endif  // SRC_RANDOM_H_
