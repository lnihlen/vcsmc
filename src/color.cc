#include "color.h"

#include <unordered_map>

// include generated color table file
#include "color_table.cc"

namespace vcsmc {

// static
uint32 Color::AtariColorToABGR(uint8 atari_color) {
  return kAtariNTSCABGRColorTable[atari_color / 2];
}

// static
const float* Color::AtariColorToLab(uint8 atari_color) {
  return kAtariNTSCLabColorTable[atari_color * 2];
}

}  // namespace vcsmc
