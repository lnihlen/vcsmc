#ifndef SRC_SCAN_LINE_H_
#define SRC_SCAN_LINE_H_

#include <vector>

#include "types.h"

namespace vcsmc {

class ColuStrip;
class State;
namespace op {
  class OpCode;
}

// A ScanLine represents a program that will attempt to render one ColuStrip.
// It has a starting State, which is the state of the VCS on entry to the
// ScanLine. It also has a target ColuStrip, which is the ColuStrip this
// ScanLine is attempting to render with minimum error. It has a list of
// (OpCode, new state) pairs representing the program changes to the TIA state
// machine. It can evaluate this to produce a predicted output ColuStrip. It can
// also output its program in assembly language (and possibly later bytecode).
// It can answer questions about its size in bytecode bytes as well as its
// length in Color Clocks or CPU cycles.
class ScanLine {
 public:
  // Makes a copy of |entry_state|.
  ScanLine(const std::unique_ptr<State>& entry_state);
  // Advance from initial state through final and produce predicted output color
  // strip.
  std::unique_ptr<ColuStrip> Simulate();

  const std::unique_ptr<State>& final_state() const {
    return *(states_.rbegin());
  }

  // Takes ownership of |opcode|, adds resultant state to |states_|.
  void AddOperation(std::unique_ptr<op::OpCode> opcode);

  // Returns the assembly language output of our opcodes.
  const std::string Assemble() const;

 private:
  // |states_| and |opcodes_| are expected to be interleaved, with states
  // bookending opcodes, as in:
  // state_0 | opcode_0 | state_1 | opcode_1 | .... | opcode_n-1 | state_n
  // state_0 is the entry state of the ScanLine and represents the state before
  // opcode_0 is executed. state_1 represents the state of the system after
  // opcode_0 was executed, and so on until state_n, which represents the final
  // state of the system.
  std::vector<std::unique_ptr<State>> states_;
  std::vector<std::unique_ptr<op::OpCode>> opcodes_;

  // going to need to start tracking these most likely
  // uint32 total_cycles_;
  // uint32 total_size_;
};

}  // namespace vcsmc

#endif  // SRC_SCAN_LINE_H_
