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
  float total_error = 0.0f;
  for (uint32 i = 0; i < kFrameWidthPixels; ++i)
    total_error += strip->Distance(i, palette->colu(fit_[i]));
  return total_error;
}

void ShapeFit::AppendSpecs(std::vector<Spec>* specs) {
  for (size_t i = 0; i < specs_.size(); ++i)
    specs->push_back(specs_[i]);
}

uint32 ShapeFit::UpdatePixelsMatched(const Palette* palette) {
  pixels_matched_ = 0;
  for (uint32 i = 0; i < kFrameWidthPixels; ++i) {
    if (fit_[i] == palette->colu_class(i))
      ++pixels_matched_;
  }
  return pixels_matched_;
}


// PlayfieldShapeFit ==========================================================

PlayfieldShapeFit::PlayfieldShapeFit(const uint8* initial_fit, uint8 colu_class)
    : ShapeFit(initial_fit, colu_class) {
}

uint32 PlayfieldShapeFit::DoFit(const Palette* palette, uint32 row_time) {
  // Simple majority-rules playfield fitting. If 3 or 4 of the pixels match the
  // playfield class then we render those pixels in the pf color.
  uint8 playfield_matched[kFrameWidthPixels / 4];
  std::memset(playfield_matched, 0, kFrameWidthPixels / 4);
  for (uint32 i = 0; i < kFrameWidthPixels; ++i) {
    if (fit_[i] == colu_class_)
      ++playfield_matched[i / 4];
  }

  // PF0 D4 through D7 left to right.
  uint8 pf0 = 0;
  for (uint32 i = 0; i < 4; ++i) {
    pf0 = pf0 >> 1;
    if (playfield_matched[i] >= 3) {
      pf0 = pf0 | 0x80;
      for (uint32 j = i * 4; j < (i * 4) + 4; ++j)
        fit_[j] = colu_class_;
    }
  }
  specs_.push_back(Spec(TIA::PF0, pf0, Range(row_time + 68, row_time + 84)));

  // PF1 D7 through D0 left to right.
  uint8 pf1 = 0;
  for (uint32 i = 4; i < 12; ++i) {
    pf1 = pf1 << 1;
    if (playfield_matched[i] >= 3) {
      pf1 = pf1 | 0x01;
      for (uint32 j = i * 4; j < (i * 4) + 4; ++j)
        fit_[j] = colu_class_;
    }
  }
  specs_.push_back(Spec(TIA::PF1, pf1, Range(row_time + 84, row_time + 116)));

  // PF2 D0 through D7 left to right.
  uint8 pf2 = 0;
  for (uint32 i = 12; i < 20; ++i) {
    pf2 = pf2 >> 1;
    if (playfield_matched[i] >= 3) {
      pf2 = pf2 | 0x80;
      for (uint32 j = i * 4; j < (i * 4) + 4; ++j)
        fit_[j] = colu_class_;
    }
  }
  specs_.push_back(Spec(TIA::PF2, pf2, Range(row_time + 116, row_time + 148)));

  // PF0 D4 through D7 left to right.
  pf0 = 0;
  for (uint32 i = 20; i < 24; ++i) {
    pf0 = pf0 >> 1;
    if (playfield_matched[i] >= 3) {
      pf0 = pf0 | 0x80;
      for (uint32 j = i * 4; j < (i * 4) + 4; ++j)
        fit_[j] = colu_class_;
    }
  }
  specs_.push_back(Spec(TIA::PF0, pf0, Range(row_time + 148, row_time + 164)));

  // PF1 D7 through D0 left to right.
  pf1 = 0;
  for (uint32 i = 24; i < 32; ++i) {
    pf1 = pf1 << 1;
    if (playfield_matched[i] >= 3) {
      pf1 = pf1 | 0x01;
      for (uint32 j = i * 4; j < (i * 4) + 4; ++j)
        fit_[j] = colu_class_;
    }
  }
  specs_.push_back(Spec(TIA::PF1, pf1, Range(row_time + 164, row_time + 196)));

  // PF2 D0 through D7 left to right.
  pf2 = 0;
  for (uint32 i = 32; i < 40; ++i) {
    pf2 = pf2 >> 1;
    if (playfield_matched[i] >= 3) {
      pf2 = pf2 | 0x80;
      for (uint32 j = i * 4; j < (i * 4) + 4; ++j)
        fit_[j] = colu_class_;
    }
  }
  specs_.push_back(Spec(TIA::PF2, pf2, Range(row_time + 196, row_time + 228)));

  return UpdatePixelsMatched(palette);
}


// PlayerShapeFit =============================================================

PlayerShapeFit::PlayerShapeFit(const uint8* initial_fit,
                               uint8 colu_class,
                               uint8 player_number,
                               uint32 last_player_position,
                               uint8 last_player_pattern)
    : ShapeFit(initial_fit, colu_class) {
}

uint32 PlayerShapeFit::DoFit(const Palette* palette, uint32 row_time) {
  return 0;
}

}  // namespace vcsmc
