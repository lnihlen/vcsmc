#include "image_fitter.h"

#include <memory>

#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "image.h"
#include "palette.h"
#include "pixel_strip.h"
#include "random.h"
#include "range.h"
#include "shape_fit.h"
#include "spec.h"

namespace vcsmc {

ImageFitter::ImageFitter(std::unique_ptr<Image> image)
    : image_(std::move(image)) {
}

std::unique_ptr<std::vector<Spec>> ImageFitter::Fit() {
  Random random;

  std::unique_ptr<CLCommandQueue> queue(CLDeviceContext::MakeCommandQueue());
  if (!queue)
    return NULL;

  if (!image_->CopyToDevice(queue.get()))
    return NULL;

  std::unique_ptr<std::vector<Spec>> specs(new std::vector<Spec>);

  // Might make sense to make a Player object to hold state. Also might make
  // sense to give each player in a frame a unique player ID and let the backend
  // decide which of two players will be less work to use.


  // We start both players at the left edge of the screen, worth consideration
  // about maybe a more optimal placement. Or maybe first line determines
  // placement? This all ties back to startup state calculation. Maybe time
  // to re-use State object which adds a method Apply() which just modifies
  // itself in place, for use as a player state container?
  uint32 p0_position = 0;
  uint32 p1_position = 0;

  for (uint32 row = 0; row < kFrameHeightPixels; ++row) {
    uint32 row_time = row * kScanLineWidthClocks;
    // First we fit colors. We can paint at most 4 colors per line, although it
    // is possible to paint more colors we limit it to 4 per now. So we build
    // palettes of each color and then use the minimum-error palette of the 4
    // for shape fitting below.
    std::unique_ptr<PixelStrip> strip = image_->GetPixelStrip(row);
    strip->BuildDistances(queue.get());
    strip->BuildPalettes(4, &random);
    const Palette* palette = strip->palette(1);
    float least_palette_error = palette->error();
    for (uint32 i = 2; i <= 4; ++i) {
      if (strip->palette(i)->error() < least_palette_error) {
        palette = strip->palette(i);
        least_palette_error = palette->error();
      }
    }

    // Always make the most frequent color the background color, as it requires
    // the least amount of computational work. While it is possible to imagine
    // scenarios where this might not be true it has intuitive appeal. The spec
    // for the background color must land by the time we start rendering pixels,
    // and should not impact the previous scan line so it must exist within
    // HBlank only.
    specs->push_back(Spec(TIA::COLUBK, palette->colu(0),
        Range(row_time + kHBlankWidthClocks,
            row_time + kHBlankWidthClocks + kFrameWidthPixels)));

    // If our least-error palette contains more than the background color we
    // proceed with further shape fitting, otherwise we just turn all other
    // graphics objects off.
    if (palette->num_colus() > 1) {
      // Our initial fit is just the background color fill, which is always
      // color class 0.
      uint8 current_fit[kFrameWidthPixels];
      std::memset(current_fit, 0, kFrameWidthPixels);

      // Examine each remaining color to find the minimum error playfield fit.
      std::unique_ptr<ShapeFit> best_playfield_fit(new PlayfieldShapeFit(
          current_fit, 1));
      uint32 best_playfield_fit_score = best_playfield_fit->DoFit(palette,
          row_time);
      for (uint32 i = 2; i < palette->num_colus(); ++i) {
        std::unique_ptr<ShapeFit> playfield_fit(new PlayfieldShapeFit(
            current_fit, i));
        uint32 fit_score = playfield_fit->DoFit(palette, row_time);
        if (fit_score > best_playfield_fit_score) {
          best_playfield_fit = std::move(playfield_fit);
          best_playfield_fit_score = fit_score;
        }
      }
    } else {
      // Simply require all other elements to draw in the same color as the
      // background.
      specs->push_back(Spec(TIA::COLUPF, palette->colu(0),
        Range(row_time + kHBlankWidthClocks,
            row_time + kHBlankWidthClocks + kFrameWidthPixels)));
      specs->push_back(Spec(TIA::COLUP0, palette->colu(0),
        Range(row_time + kHBlankWidthClocks,
            row_time + kHBlankWidthClocks + kFrameWidthPixels)));
      specs->push_back(Spec(TIA::COLUP1, palette->colu(0),
        Range(row_time + kHBlankWidthClocks,
            row_time + kHBlankWidthClocks + kFrameWidthPixels)));
    }

    row_time += kFrameWidthPixels;
  }

  return specs;
}


}  // namespace vcsmc
