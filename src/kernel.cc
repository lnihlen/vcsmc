#include "kernel.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include "background_color_strategy.h"
#include "constants.h"
#include "do_nothing_strategy.h"
#include "image.h"
#include "opcode.h"
#include "pixel_strip.h"
#include "scan_line.h"
#include "state.h"
#include "tiff_image_file.h"

namespace vcsmc {

Kernel::Kernel(std::unique_ptr<Image> target_image)
    : target_image_(std::move(target_image)),
      output_image_(new Image(kFrameWidthPixels * 2, kFrameHeightPixels)) {
}

void Kernel::Fit() {
  scan_lines_.reserve(kFrameHeightPixels);

  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {
    std::unique_ptr<State> entry_state = EntryStateForLine(i);
    std::unique_ptr<PixelStrip> target_strip = target_image_->GetPixelStrip(i);

    //========== Do Nothing!

    DoNothingStrategy do_nothing;
    std::unique_ptr<ScanLine> do_nothing_scan_line = do_nothing.Fit(
        target_strip.get(), entry_state.get());
    std::unique_ptr<PixelStrip> do_nothing_strip =
        do_nothing_scan_line->Simulate();
    double do_nothing_error = do_nothing_strip->DistanceFrom(
        target_strip.get());

    if (do_nothing_error == 0) {
      output_image_->SetStrip(i, do_nothing_strip.get());
      scan_lines_.push_back(std::move(do_nothing_scan_line));
      continue;
    }

    //========== Set background color only.

    BackgroundColorStrategy bg_color;
    std::unique_ptr<ScanLine> bg_color_scan_line = bg_color.Fit(
      target_strip.get(), entry_state.get());
    std::unique_ptr<PixelStrip> bg_color_strip =
      bg_color_scan_line->Simulate();
    double bg_color_error = bg_color_strip->DistanceFrom(
        target_strip.get());

    //========== Pick minimum error result.

    if (bg_color_error < do_nothing_error) {
      output_image_->SetStrip(i, bg_color_strip.get());
      scan_lines_.push_back(std::move(bg_color_scan_line));
    } else {
      output_image_->SetStrip(i, do_nothing_strip.get());
      scan_lines_.push_back(std::move(do_nothing_scan_line));
    }
  }
}

void Kernel::Save() {
  // Write out assembler.
  std::ofstream of("kernel.asm");
  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {
    of << std::endl << "; -- scan line: " << i << std::endl;
    of << scan_lines_[i]->Assemble();
  }

  // Write out predicted image.
  TiffImageFile tiff("kernel.tiff");
  tiff.Save(output_image_.get());
}

std::unique_ptr<State> Kernel::EntryStateForLine(uint32 line) {
  // Update entry state to final state of last line + time to new line.
  if (line > 0) {
    assert(line - 1 < scan_lines_.size());
    uint32 delta = (line * kScanLineWidthClocks) -
        scan_lines_[line - 1]->final_state()->color_clocks();
    return scan_lines_[line - 1]->final_state()->AdvanceTime(delta);
  }
  return std::unique_ptr<State>(new State());
}

}  // namespace vcsmc
