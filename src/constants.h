#ifndef SRC_CONSTANTS_H_
#define SRC_CONSTANTS_H_

#include "types.h"

namespace vcsmc {

// Terminology:
// a Pixel is an actual renderable element, on screen.
// a Clock is a Color_Clock, or the time required to render one pixel
// a Cycle is a CPU cycle, on the VCS is 3 Clocks.
const uint32 kFrameWidthPixels = 160;
const uint32 kHBlankWidthClocks = 68;
const uint32 kScanLineWidthClocks = kFrameWidthPixels + kHBlankWidthPixels;
const uint32 kColorClocksPerCPUCycle = 3;
const uint32 kScanLineWidthCycles = kScanLineWidthClocks /
                                    kColorClocksPerCPUCycle;

// The PIA provides 128 bytes of RAM starting at zero page address $80. The
// stack is also used for this, it grows down from $ff. We reserve only 1 byte
// for the stack.
const uint32 kNumberOfBytesOfRAM = 127;

}  // namespace vcsmc

#endif  // SRC_CONSTANTS_H_
