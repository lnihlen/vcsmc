#include "do_nothing_strategy.h"

#include <cassert>

#include "opcode.h"  // doing nothing requires knowledge of many things.
#include "scan_line.h"
#include "state.h"

namespace vcsmc {

std::unique_ptr<ScanLine> DoNothingStrategy::Fit(PixelStrip* target_strip,
                                                 State* entry_state) {
  // Start scanline on a color-clock scanline boundary.
  assert((entry_state->color_clocks() % kScanLineWidthClocks) == 0);
  // Doing nothing means the state will be the same at exit that it was at
  // entry. Advance the time to the new scanline.
  return std::unique_ptr<ScanLine>(new ScanLine(entry_state));
}

}  // namespace vcsmc
