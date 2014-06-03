#include "kernel.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include "background_color_strategy.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "colu_strip.h"
#include "constants.h"
#include "do_nothing_strategy.h"
#include "image.h"
#include "opcode.h"
#include "pallette.h"
#include "pixel_strip.h"
#include "playfield_strategy.h"
#include "random.h"
#include "schedule.h"
#include "state.h"
#include "tiff_image_file.h"

namespace vcsmc {

Kernel::Kernel(std::unique_ptr<Image> target_image)
    : target_image_(std::move(target_image)),
      schedule_(new Schedule()),
      output_image_(new Image(kFrameWidthPixels * 2, kFrameHeightPixels)) {
}

void Kernel::Fit() {
  Random random;

  std::unique_ptr<CLCommandQueue> queue(CLDeviceContext::MakeCommandQueue());
  if (!queue)
    return;

  if (!target_image_->CopyToDevice(queue.get()))
    return;

  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {

    std::unique_ptr<PixelStrip> target_strip = target_image_->GetPixelStrip(i);
    target_strip->BuildDistances(queue.get());
    target_strip->BuildPallettes(2, &random);

    //========== Do Nothing!

    DoNothingStrategy do_nothing;
    std::unique_ptr<Schedule> do_nothing_schedule = do_nothing.Fit(
        target_strip.get(), schedule_.get());
    std::unique_ptr<ColuStrip> do_nothing_strip =
        do_nothing_schedule->Simulate(i);
    float do_nothing_error = target_strip->DistanceFrom(do_nothing_strip.get());

    //========== Set background color only.

    BackgroundColorStrategy bg_color;
    std::unique_ptr<Schedule> bg_color_schedule = bg_color.Fit(
        target_strip.get(), schedule_.get());
    std::unique_ptr<ColuStrip> bg_color_strip = bg_color_schedule->Simulate(i);
    float bg_color_error = target_strip->DistanceFrom(bg_color_strip.get());

    //========== Playfield Fitting.

    PlayfieldStrategy pf_strategy;
    std::unique_ptr<Schedule> pf_schedule = pf_strategy.Fit(
        target_strip.get(), schedule_.get());
    std::unique_ptr<ColuStrip> pf_color_strip = pf_schedule->Simulate(i);
    float pf_error = target_strip->DistanceFrom(pf_color_strip.get());

    //========== Pick minimum error result.

    std::cout << "row: " << i
              << " do nothing error: " << do_nothing_error
              << " bg color error: " << bg_color_error
              << " pf error: " << pf_error
              << std::endl;

    // Yes I know we need to sort.
    if (pf_error < bg_color_error) {
      output_image_->SetStrip(i, pf_color_strip.get());
      schedule_.swap(pf_schedule);
    } else if (bg_color_error < do_nothing_error) {
      output_image_->SetStrip(i, bg_color_strip.get());
      schedule_.swap(bg_color_schedule);
    } else {
      output_image_->SetStrip(i, do_nothing_strip.get());
      schedule_.swap(do_nothing_schedule);
    }
  }
}

void Kernel::Save() {
  // Write out assembler.
  std::ofstream of("kernel.asm");
  of << schedule_->Assemble();

  // Write out predicted image.
  TiffImageFile tiff("kernel.tiff");
  tiff.Save(output_image_.get());
}

}  // namespace vcsmc
