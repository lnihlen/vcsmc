#include "shape_fit.h"

#include "palette.h"
#include "pixel_strip.h"

namespace vcsmc {

ShapeFit::ShapeFit(const uint8* initial_fit, uint8 colu_class)
    : colu_class_(colu_class),
      pixels_matched_(0) {
  std::memcpy(fit_, initial_fit, kFrameWidthPixels);
}

float ShapeFit::ComputeTotalError(
    const PixelStrip* strip, const Palette* palette) {
  return 0.0f;
}


// PlayfieldShapeFit ==========================================================

PlayfieldShapeFit::PlayfieldShapeFit(const uint8* initial_fit, uint8 colu_class)
    : ShapeFit(initial_fit, colu_class) {
}


// PlayerShapeFit =============================================================

}  // namespace vcsmc
