#include "frame.h"

#include <cassert>
#include <cstring>

#include "color.h"
#include "colu_strip.h"
#include "constants.h"
#include "image.h"

namespace vcsmc {

Frame::Frame()
    : colu_(new uint8[kFrameSizeBytes]) {
  std::memset(colu_.get(), 0, kFrameSizeBytes);
}

Frame::Frame(Image* image)
    : colu_(new uint8[kFrameSizeBytes]) {
  assert(image->width() == kFrameWidthPixels);
  assert(image->height() == kFrameHeightPixels);

  uint32 offset = 0;
  for (uint32 i = 0; i < kFrameHeightPixels; ++i) {
    for (uint32 j = 0; j < kFrameWidthPixels; ++j) {
      colu_[offset] = Color::ABGRToAtariColor(image->GetPixel(j, i));
      ++offset;
    }
  }
}

void Frame::SetColor(uint32 offset, uint8 color) {
  assert(offset < kFrameSizeBytes);
  colu_[offset] = color;
}

void Frame::SetStrip(ColuStrip* strip, uint32 row) {
  uint32 offset = row * kFrameWidthPixels;
  std::memcpy(colu_.get() + offset, strip->colus(), kFrameWidthPixels);
}

std::unique_ptr<Image> Frame::ToImage() const {
  std::unique_ptr<Image> image(new Image(kFrameWidthPixels,
                                         kFrameHeightPixels));
  uint32 offset = 0;
  for (uint32 i = 0; i < kFrameWidthPixels; ++i) {
    for (uint32 j = 0; j < kFrameWidthPixels; ++j) {
      uint32 abgr = Color::AtariColorToABGR(colu_[offset]);
      image->SetPixel(j, i, abgr);
      ++offset;
    }
  }

  return image;
}

}  // namespace vcsmc
