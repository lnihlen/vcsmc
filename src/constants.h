#ifndef SRC_CONSTANTS_H_
#define SRC_CONSTANTS_H_

#include "types.h"

namespace vcsmc {

// Terminology:
// a Pixel is an actual renderable element, on screen.
// a Clock is a Color_Clock, or the time required to render one pixel
// a Cycle is a CPU cycle, on the VCS is 3 Clocks.
const uint32 kFrameWidthPixels = 160;
const uint32 kFrameHeightPixels = 240;
const uint32 kFrameSizeBytes = kFrameWidthPixels * kFrameHeightPixels;
const uint32 kHBlankWidthClocks = 68;
const uint32 kScanLineWidthClocks = kFrameWidthPixels + kHBlankWidthClocks;
const uint32 kColorClocksPerCPUCycle = 3;
const uint32 kScanLineWidthCycles = kScanLineWidthClocks /
                                    kColorClocksPerCPUCycle;

}  // namespace vcsmc

#endif  // SRC_CONSTANTS_H_
