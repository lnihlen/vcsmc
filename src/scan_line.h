#ifndef SRC_SCAN_LINE_H_
#define SRC_SCAN_LINE_H_

#include <vector>

#include "colu_strip.h"
#include "opcodes.h"
#include "state.h"
#include "types.h"

namespace vcsmc {

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
  ScanLine(const ColuStrip& target_strip);
  ~ScanLine();

 private:
  ColuStrip target_strip_;
  // Owning vector of OpCodes in time order.
  std::vector<OpCode*> opcodes_;
  std::vector<State> states_;
};

}  // namespace vcsmc

#endif  // SRC_SCAN_LINE_H_
