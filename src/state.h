#ifndef SRC_STATE_H_
#define SRC_STATE_H_

#include <array>
#include <cassert>

#include "codon.h"
#include "constants.h"
#include "snippet.h"
#include "types.h"

namespace vcsmc {

// Tracks current state of TIA and registers. Used in sequencing codons into
// opcodes.
class State {
 public:
  State(uint32 current_time = 0);

  // Given an individual Codon, use the current State to most efficiently
  // Translate the Codon into 6502 bytecode.
  Snippet Translate(Codon codon) const;

  // Update all internal state to reflect the simulated execution of |snippet|,
  // while appending the bytecode in snippet to the |bytecode| array. 
  void Apply(const Snippet& snippet, uint8* bytecode);

  uint8* tia() { return tia_.data(); }
  uint8* registers() { return registers_.data(); }

  // Accessors mostly used for testing.
  uint32* register_last_used() { return register_last_used_.data(); }
  uint32 current_time() { return current_time_; }
  void set_current_time(uint32 time) { current_time_ = time; }

 private:
  std::array<uint8, TIA_COUNT> tia_;
  std::array<uint8, REGISTER_COUNT> registers_;
  std::array<uint32, REGISTER_COUNT> register_last_used_;
  uint32 current_time_;
};

}

#endif  // SRC_STATE_H_
