#include "color.h"

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "constants.h"

// include generated color table file
#include "auto/color_table.cc"

namespace vcsmc {

// static
const uint32 Color::AtariColorToABGR(uint8 atari_color) {
  return kAtariNTSCABGRColorTable[atari_color / 2];
}

// static
const float* Color::AtariColorToLab(uint8 atari_color) {
  return kAtariNTSCLabColorTable + (atari_color * 2);
}

}  // namespace vcsmc
