#include "scan_line.h"

#include <iostream>

#include "histogram.h"
#include "opcode.h"
#include "pixel_strip.h"
#include "state.h"

namespace vcsmc {

ScanLine::ScanLine(State* entry_state) {
  states_.push_back(entry_state->Clone());
}

std::unique_ptr<PixelStrip> ScanLine::Simulate() {
  std::unique_ptr<PixelStrip> pixel_strip(
      new PixelStrip(kFrameWidthPixels * 2));
  for (uint32 i = 0; i < states_.size(); ++i) {
    uint32 until = states_[0]->color_clocks() + kScanLineWidthClocks;
    if (i + 1 < states_.size()) {
      until = states_[i + 1]->color_clocks();
    }
    states_[i]->PaintInto(pixel_strip.get(), until);
  }
  return pixel_strip;
}

void ScanLine::AddOperation(std::unique_ptr<op::OpCode> opcode) {
  // Generate new state as a result of this opcode transforming last state.
  states_.push_back(opcode->Transform(final_state()));
  opcodes_.push_back(std::move(opcode));
}

const std::string ScanLine::Assemble() const {
  std::string assembly;
  for (uint32 i = 0; i < opcodes_.size(); ++i) {
    assembly += "  " + opcodes_[i]->assembler() + "\n";
  }
  // not always needed!
  assembly += "  sta WSYNC\n";
  return assembly;
}

}  // namespace vcsmc
