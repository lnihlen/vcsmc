#ifndef SRC_STATE_H_
#define SRC_STATE_H_

#include <array>

#include "constants.h"
#include "types.h"

namespace vcsmc {

// Tracks current state of TIA and registers. Used in sequencing codons into
// opcodes.
class State {
 public:
  State();

 private:
  std::array<uint8, TIA_COUNT> tia_state_;
  std::array<unit8, REGISTER_COUNT> register_state_;
};

}

#endif  // SRC_STATE_H_
