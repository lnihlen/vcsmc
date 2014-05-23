#include "playfield_strategy.h"

#include <cassert>
#include <vector>

#include "color.h"
#include "opcode.h"
#include "pallette.h"
#include "pixel_strip.h"
#include "scan_line.h"
#include "state.h"
#include "types.h"

namespace vcsmc {

std::unique_ptr<ScanLine> PlayfieldStrategy::Fit(
    PixelStrip* target_strip, State* entry_state) {
  assert((target_strip->width() % 8) == 0);

  const Pallette* pallette = target_strip->pallette(2);
  assert(pallette);

  uint8 colubk = pallette->colu(0);
  uint8 colupf = pallette->colu(1);

  // Use first 40 bits to represent playfield. Most significant bit (bit 39)
  // represents the leftmost playfield bit.
  uint64 playfield = 0;
  for (uint32 i = 0; i < target_strip->width(); i += 8) {
    // Compare errors by modeling this bit of playfield as BK or PF color.
    float bk_error = 0.0;
    float pf_error = 0.0;
    for (uint32 offset = i; offset < i + 8; ++offset) {
      bk_error += target_strip->distance(offset, colubk);
      pf_error += target_strip->distance(offset, colupf);
    }

    playfield = playfield << 1;
    playfield = playfield | (pf_error < bk_error ? 1 : 0);
  }

  uint64 bitmask = 0x8000000000;  // 1 << 39
  // First 4 pixels are PF0 D4 through D7 left to right.
  uint8 pf0_left = 0;
  for (uint32 i = 0; i < 4; ++i) {
    pf0_left = pf0_left >> 1;
    pf0_left = pf0_left | (playfield & bitmask ? 0x80 : 0x00);
    bitmask = bitmask >> 1;
  }

  // Next 8 pixels are PF1 D7 through D0 left to right.
  uint8 pf1_left = 0;
  for (uint32 i = 0; i < 8; ++i) {
    pf1_left = pf1_left << 1;
    pf1_left = pf1_left | (playfield & bitmask ? 0x01 : 0x00);
    bitmask = bitmask >> 1;
  }

  // Next 8 pixels are PF2 D0 through D7 left to right.
  uint8 pf2_left = 0;
  for (uint32 i = 0; i < 8; ++i) {
    pf2_left = pf2_left >> 1;
    pf2_left = pf2_left | (playfield & bitmask ? 0x80 : 0x00);
    bitmask = bitmask >> 1;
  }

  // Next 4 pixels are PF0 D7 through D4.
  uint8 pf0_right = 0;
  for (uint32 i = 0; i < 4; ++i) {
    pf0_right = pf0_right >> 1;
    pf0_right = pf0_right | (playfield & bitmask ? 0x80 : 0x00);
    bitmask = bitmask >> 1;
  }

  // Next 8 pixels are PF1 D0 through D7.
  uint8 pf1_right = 0;
  for (uint32 i = 0; i < 8; ++i) {
    pf1_right = pf1_right << 1;
    pf1_right = pf1_right | (playfield & bitmask ? 0x01 : 0x00);
    bitmask = bitmask >> 1;
  }

  // Last 8 pixels are PF2 D7 through D0.
  uint8 pf2_right = 0;
  for (uint32 i = 0; i < 8; ++i) {
    pf2_right = pf2_right >> 1;
    pf2_right = pf2_right | (playfield & bitmask ? 0x80 : 0x00);
    bitmask = bitmask >> 1;
  }

  // Usage schedule for playfield registers, in color clocks:
  // PF0 68 up until 84
  // PF1 84 - 116
  // PF2 116 - 148
  // PF0 148 - 164
  // PF1 164 - 196
  // PF2 196 - 228

  // Action       | start time (cc) |
  // -------------+-----------------+
  // lda #colorbk |         0       |
  // sta COLUBK   |         6       |
  // lda #colupf  |        15       |
  // sta COLUPF   |        21       |
  // lda #pf0     |        30       |
  // sta PF0      |        36       |
  // lda #pf1     |        45       |
  // sta PF1      |        51       |
  // lda #pf2     |        60       |
  // sta PF2      |        66       |
  // lda #pf0     |        75       |
  // nop          |        81       |
  // sta PF0      |        87       |
  // lda #pf1     |        96       |
  // nop          |       102       |
  // nop          |       108       |
  // nop          |       114       |
  // sta PF1      |       120       |
  // lda #pf2     |       129       |
  // nop          |       135       |
  // nop          |       141       |
  // nop          |       147       |
  // sta PF2      |       153       |
  // sta WSYNC    |       162       |
  std::unique_ptr<ScanLine> scan_line(new ScanLine(entry_state));
  scan_line->AddOperation(makeLDA(colubk));
  scan_line->AddOperation(makeSTA(State::TIA::COLUBK));
  scan_line->AddOperation(makeLDA(colupf));
  scan_line->AddOperation(makeSTA(State::TIA::COLUPF));
  scan_line->AddOperation(makeLDA(pf0_left));
  scan_line->AddOperation(makeSTA(State::TIA::PF0));
  scan_line->AddOperation(makeLDA(pf1_left));
  scan_line->AddOperation(makeSTA(State::TIA::PF1));
  scan_line->AddOperation(makeLDA(pf2_left));
  scan_line->AddOperation(makeSTA(State::TIA::PF2));
  scan_line->AddOperation(makeLDA(pf0_right));
  scan_line->AddOperation(makeNOP());
  scan_line->AddOperation(makeSTA(State::TIA::PF0));
  scan_line->AddOperation(makeLDA(pf1_right));
  scan_line->AddOperation(makeNOP());
  scan_line->AddOperation(makeNOP());
  scan_line->AddOperation(makeNOP());
  scan_line->AddOperation(makeSTA(State::TIA::PF1));
  scan_line->AddOperation(makeLDA(pf2_right));
  scan_line->AddOperation(makeNOP());
  scan_line->AddOperation(makeNOP());
  scan_line->AddOperation(makeNOP());
  scan_line->AddOperation(makeSTA(State::TIA::PF2));

  return scan_line;
}

}  // namespace vcsmc
