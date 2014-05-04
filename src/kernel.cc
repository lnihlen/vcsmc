#include "kernel.h"

#include <fstream>
#include <iostream>

#include "background_color_strategy.h"
#include "constants.h"
#include "do_nothing_strategy.h"
#include "image.h"
#include "opcode.h"
#include "state.h"
#include "tiff_image_file.h"

namespace vcsmc {

Kernel::Kernel(std::unique_ptr<Frame> target_frame)
    : target_frame_(std::move(target_frame)) {
}

void Kernel::Fit() {
  output_frame_.reset(new Frame());

  // We start with an empty State.
  std::unique_ptr<State> entry_state(new State());
  scan_lines_.reserve(kFrameHeightPixels);

  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {
    // Update entry state to final state of last line + time to new line.
    if (i > 0) {
      uint32 delta = (i * kScanLineWidthClocks) -
          scan_lines_[i - 1]->final_state()->color_clocks();
      entry_state = scan_lines_[i - 1]->final_state()->AdvanceTime(delta);
    }

    std::unique_ptr<ColuStrip> target_strip = target_frame_->GetStrip(i);

    //========== Do Nothing!

    DoNothingStrategy do_nothing;
    std::unique_ptr<ScanLine> do_nothing_scan_line = do_nothing.Fit(
        target_strip, entry_state);
    std::unique_ptr<ColuStrip> do_nothing_colu_strip =
        do_nothing_scan_line->Simulate();
    double do_nothing_error = do_nothing_colu_strip->DistanceFrom(target_strip);

    if (do_nothing_error == 0) {
      scan_lines_.push_back(std::move(do_nothing_scan_line));
      output_frame_->SetStrip(do_nothing_colu_strip, i);
      continue;
    }

    //========== Set background color only.

    BackgroundColorStrategy bg_color;
    std::unique_ptr<ScanLine> bg_color_scan_line = bg_color.Fit(
      target_strip, entry_state);
    std::unique_ptr<ColuStrip> bg_color_colu_strip =
      bg_color_scan_line->Simulate();
    double bg_color_error = bg_color_colu_strip->DistanceFrom(target_strip);

    //========== Pick minimum error result.

    if (bg_color_error < do_nothing_error) {
      output_frame_->SetStrip(bg_color_colu_strip, i);
      scan_lines_.push_back(std::move(bg_color_scan_line));
    } else {
      output_frame_->SetStrip(do_nothing_colu_strip, i);
      scan_lines_.push_back(std::move(do_nothing_scan_line));
    }
  }
}

void Kernel::Save() {
  // Write out assembler.
  std::ofstream of("kernel.asm");
  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {
    of << "; -- scan line: " << i << std::endl << std::endl;
    of << scan_lines_[i]->Assemble();
  }

  // Write out predicted image.
  TiffImageFile tiff("kernel.tiff");
  tiff.Save(output_frame_->ToImage());
}

}  // namespace vcsmc
