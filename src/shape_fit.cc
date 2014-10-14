#include "shape_fit.h"

#include "palette.h"
#include "pixel_strip.h"
#include "spec.h"

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

void ShapeFit::AppendSpecs(std::vector<Spec>* specs) {
  for (size_t i = 0; i < specs_.size(); ++i)
    specs->push_back(specs_[i]);
}


// PlayfieldShapeFit ==========================================================

PlayfieldShapeFit::PlayfieldShapeFit(const uint8* initial_fit, uint8 colu_class)
    : ShapeFit(initial_fit, colu_class) {
}

uint32 PlayfieldShapeFit::DoFit(const Palette* palette) {
  return 0;
}


// PlayerShapeFit =============================================================

PlayerShapeFit::PlayerShapeFit(const uint8* initial_fit,
                               uint8 colu_class,
                               uint8 player_number,
                               uint32 last_player_position,
                               uint8 last_player_pattern)
    : ShapeFit(initial_fit, colu_class) {
}

uint32 PlayerShapeFit::DoFit(const Palette* palette) {
  return 0;
}

}  // namespace vcsmc
