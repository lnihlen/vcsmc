#ifndef SRC_CONSTANTS_H_
#define SRC_CONSTANTS_H_

#include "types.h"

namespace vcsmc {

// Terminology:
// a Pixel is an actual renderable element, on screen.
// a Clock is a Color_Clock, or the time required to render one pixel
// a Cycle is a CPU cycle, on the VCS is 3 Clocks.
const uint32 kFrameWidthPixels = 160;
const uint32 kFrameHeightPixels = 180;
const uint32 kFrameSizeBytes = kFrameWidthPixels * kFrameHeightPixels;
const uint32 kHBlankWidthClocks = 68;
const uint32 kScanLineWidthClocks = kFrameWidthPixels + kHBlankWidthClocks;
const uint32 kColorClocksPerCPUCycle = 3;
const uint32 kScanLineWidthCycles = kScanLineWidthClocks /
                                    kColorClocksPerCPUCycle;
const uint32 kNTSCColors = 128;

// Defines the address and name of every register on the TIA. The ones marked
// as (strobe) are write-only and writing to them will cause new changes in
// state.
enum TIA : uint8 {
  VSYNC  = 0x00,  // vertical sync set-clear
  VBLANK = 0x01,  // vertical blank set-clear
  WSYNC  = 0x02,  // (strobe) wait for leading edge of horizontal blank
  RSYNC  = 0x03,  // (strobe) reset horizontal sync counter
  NUSIZ0 = 0x04,  // number-size player-missile 0
  NUSIZ1 = 0x05,  // number-size player-missile 1
  COLUP0 = 0x06,  // color-lum player 0
  COLUP1 = 0x07,  // colur-lum player 1
  COLUPF = 0x08,  // colur-lum playfield
  COLUBK = 0x09,  // colur-lum background
  CTRLPF = 0x0a,  // control playfield ball size & collisions
  REFP0  = 0x0b,  // reflect player 0
  REFP1  = 0x0c,  // reflect player 1
  PF0    = 0x0d,  // playfield register byte 0
  PF1    = 0x0e,  // playfield register byte 1
  PF2    = 0x0f,  // playfield register byte 2
  RESP0  = 0x10,  // (strobe) reset player 0
  RESP1  = 0x11,  // (strobe) reset player 1
  RESM0  = 0x12,  // (strobe) reset missile 0
  RESM1  = 0x13,  // (strobe) reset missile 1
  RESBL  = 0x14,  // (strobe) reset ball
  AUDC0  = 0x15,  // audio control 0
  AUDC1  = 0x16,  // audio control 1
  AUDF0  = 0x17,  // audio frequency 0
  AUDF1  = 0x18,  // audio frequency 1
  AUDV0  = 0x19,  // audio volume 0
  AUDV1  = 0x1a,  // audio volume 1
  GRP0   = 0x1b,  // graphics player 0
  GRP1   = 0x1c,  // graphics player 1
  ENAM0  = 0x1d,  // graphics (enable) missile 0
  ENAM1  = 0x1e,  // graphics (enable) missile 1
  ENABL  = 0x1f,  // graphics (enable) ball
  HMP0   = 0x20,  // horizontal motion player 0
  HMP1   = 0x21,  // horizontal motion player 1
  HMM0   = 0x22,  // horizontal motion missile 0
  HMM1   = 0x23,  // horizontal motion missile 1
  HMBL   = 0x24,  // horizontal motion ball
  VDELP0 = 0x25,  // vertical delay player 0
  VDELP1 = 0x26,  // vertical delay player 1
  VDELBL = 0x27,  // vertical delay ball
  RESMP0 = 0x28,  // reset missile 0 to player 0
  RESMP1 = 0x29,  // reset missile 1 to player 1
  HMOVE  = 0x2a,  // (strobe) apply horizontal motion
  HMCLR  = 0x2b,  // (strobe) clear horizontal motion registers
  CXCLR  = 0x2c,  // (strobe) clear collision latches
  TIA_COUNT  = 0x2d
};

enum Register : uint8 {
  A = 0,
  X = 1,
  Y = 2,
  REGISTER_COUNT = 3
};

}  // namespace vcsmc

#endif  // SRC_CONSTANTS_H_
