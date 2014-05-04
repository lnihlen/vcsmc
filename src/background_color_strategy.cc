#include "background_color_strategy.h"

#include "colu_strip.h"
#include "histogram.h"
#include "opcode.h"
#include "scan_line.h"
#include "state.h"

namespace vcsmc {

std::unique_ptr<ScanLine> BackgroundColorStrategy::Fit(
    const std::unique_ptr<ColuStrip>& target_strip,
    const std::unique_ptr<State>& entry_state) {
  // Histogram the colors in the target_strip.
  Histogram histo(target_strip);
  // Choose most frequent color.
  uint8 colubk = histo.colu(0);
  // Now figure out most efficient means to pack it in to a new ScanLine.
  std::unique_ptr<ScanLine> scan_line(new ScanLine(entry_state));
  // Is bg color currently what we need?
  if (entry_state->tia(State::TIA::COLUBK) != colubk) {
    //** To me it seems this kind of thinking may best fit bcak in ScanLine.
    // Find if a register already has this value
    if (entry_state->a() == colubk) {
      scan_line->AddOperation(std::unique_ptr<op::OpCode>(
          new op::STA(State::TIA::COLUBK)));
    } else if (entry_state->x() == colubk) {
      scan_line->AddOperation(std::unique_ptr<op::OpCode>(
          new op::STX(State::TIA::COLUBK)));
    } else if (entry_state->y() == colubk) {
      scan_line->AddOperation(std::unique_ptr<op::OpCode>(
          new op::STY(State::TIA::COLUBK)));
    } else {
      // register packing is a whole 'nother trip.
      State::Register reg = static_cast<State::Register>(
          (entry_state->color_clocks() / kScanLineWidthClocks) %
          State::Register::REGISTER_COUNT);
      scan_line->AddOperation(std::unique_ptr<op::OpCode>(
          new op::LoadImmediate(colubk, reg)));
      scan_line->AddOperation(std::unique_ptr<op::OpCode>(
          new op::StoreZeroPage(State::TIA::COLUBK, reg)));
    }
  }
  return scan_line;
}

}  // namespace vcsmc
