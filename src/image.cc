#include "atari_ntsc_abgr_color_table.h"
#include "image.h"
#include "constants.h"

namespace vcsmc {

Image::Image(uint32 width, uint32 height)
    : width_(width),
      height_(height),
      pixels_(new uint32[width * height]) {
}

Image::Image(const uint8* atari_colors)
    : Image(kTargetFrameWidthPixels, kFrameHeightPixels) {
  uint32* pixel = pixels_.get();
  const uint8* atari_color = atari_colors;
  for (size_t i = 0; i < kTargetFrameWidthPixels * kFrameHeightPixels; i += 2) {
    uint32 color = *atari_color < 128 ? kAtariNtscAbgrColorTable[*atari_color] :
        0xff00ff00;
    *pixel = color;
    ++pixel;
    *pixel = color;
    ++pixel;
    atari_color += 2;
  }
}

Image::~Image() {
}

}  // namespace vcsmc
