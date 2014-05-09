#include "playfield_strategy.h"

#include <vector>

#include "pixel_strip.h"
#include "state.h"
#include "types.h"

namespace vcsmc {

// Usage schedule for playfield registers, in color clocks:
// PF0 68 up until 84
// PF1 84 - 116
// PF2 116 - 148
// PF0 148 - 164
// PF1 164 - 196
// PF2 196 - 228

// Action       | start time |
// -------------+------------|
// lda #colorbk |     0      |
// sta COLUBK   |     2      |
// lda #colupf  |     5      |
// sta COLUPF   |     7      |
// lda #pf0     |    10      |
// sta PF0      |    
virtual std::unique_ptr<ScanLine> Fit(PixelStrip* target_strip,
                                      State* entry_state) {
  assert((target_strip->width() % 8) == 0);

  Histogram* histo = target_strip->histo();
  assert(histo);

  uint8 colubk = histo->colu(0);
  uint8 colupf = histo->colu(1);
  uint32 color_bk = Color::AtariColorToABGR(colubk);
  uint32 color_pf = Color::AtariColorToABGR(colupf);

  // Use first 40 bits to represent playfield.
  uint64 playfield = 0;
  for (uint32 i = 0; i < target_strip->width(); i += 8) {
    // Compare errors by modeling this bit of playfield as BK or PF color.
    double bk_error = 0.0;
    double pf_error = 0.0;
    for (uint32 offset = i; offset < i + 8; ++offset) {
      uint32 pixel = pixel_strip->color(offset);
      bk_error += Color::CartesianDistanceSquaredABGR(pixel, color_bk);
      pf_error += Color::CartesianDistanceSquaredABGR(pixel, color_pf);
    }

    if (bk_error < pf_error) {
      playfield = playfield << 1;
    } else {
      playfield = (playfield | 1) << 1;
    }
  }

  uint64 bitmask = 1 << 39;
  // First 4 pixels are PF0 D7 through D4.
  uint8 pf0 = 0;
  for (uint32 i = 0; i < 4; ++i) {
    if (playfield & bitmask) {
      pf0 = (pf0 | 0x80) >> 1;
    } else {
      pf0 = pf0 >> 1;
    }
    bitmask = bitmask >> 1;
  }

  // Next 8 pixels are PF1 D0 through D7.
  uint8 pf1 = 0;
  for (uint32 i = 0; i < 8; ++i) {
    if (playfield & bitmask) {
      pf1 = (pf1 | 1) << 1;
    } else {
      pf1 = pf1 << 1;
    }
    bitmask = bitmask >> 1;
  }

  // Next 8 pixels are PF2 D7 through D0.
  uint8 pf2 = 0;
  for (uint32 i = 0; i < 8; ++i) {
    if (playfield & bitmask) {
      pf2 = (pf2 | 0x80) >> 1;
    } else {
      pf2 = pf2 >> 1;
    }
    bitmask = bitmask >> 1;
  }

  std::unique_ptr<ScanLine> scan_line(new ScanLine(entry_state));

  return scan_line;
}

}  // namespace vcsmc
