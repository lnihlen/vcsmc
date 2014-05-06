#include "background_color_strategy.h"

#include <cassert>

#include "color.h"
#include "histogram.h"
#include "opcode.h"
#include "scan_line.h"
#include "state.h"

namespace vcsmc {

std::unique_ptr<ScanLine> BackgroundColorStrategy::Fit(
    PixelStrip* target_strip, State* entry_state) {
  // Histogram the colors in the target_strip.
  Histogram histo(target_strip);

  // Choose best-fit single color.
  uint8 colubk = 0;
  double best_color_score = histo.DistanceFrom(Color::AtariColorToABGR(0));
  for (uint32 i = 2; i < 256; i += 2) {
    double color_score = histo.DistanceFrom(Color::AtariColorToABGR(i));
    if (color_score < best_color_score) {
      colubk = i;
      best_color_score = color_score;
    }
  }

  // Now figure out most efficient means to pack it in to a new ScanLine.
  std::unique_ptr<ScanLine> scan_line(new ScanLine(entry_state));

  // Is bg color currently what we need?
  if (entry_state->tia(State::TIA::COLUBK) != colubk) {
    //** To me it seems this kind of thinking may best fit back in ScanLine.
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

  assert(scan_line->final_state()->tia(State::TIA::COLUBK) == colubk);
  return scan_line;
}

}  // namespace vcsmc
