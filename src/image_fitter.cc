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
#include "spec.h"

namespace vcsmc {

ImageFitter::ImageFitter(std::unique_ptr<Image> image)
    : image_(std::move(image)) {
}

std::unique_ptr<std::vector<Spec>> ImageFitter::Fit(uint64 base_frame_time) {
  Random random;

  std::unique_ptr<CLCommandQueue> queue(CLDeviceContext::MakeCommandQueue());
  if (!queue)
    return NULL;

  if (!image_->CopyToDevice(queue.get()))
    return NULL;

  std::unique_ptr<std::vector<Spec>> specs(new std::vector<Spec>);
  uint64 row_time = base_frame_time;

  for (uint32 row = 0; row < kFrameHeightPixels; ++row) {
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

    // If our last-error palette contains more than the background color we
    // proceed with further shape fitting, otherwise we just turn all other
    // graphics objects off.
    if (palette->num_colus() > 1) {

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
