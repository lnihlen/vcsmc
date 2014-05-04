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

Frame::Frame(const std::unique_ptr<Image>& image)
    : colu_(new uint8[kFrameSizeBytes]) {
  assert(image->width() == kFrameWidthPixels);
  assert(image->height() == kFrameHeightPixels);

  uint32 offset = 0;
  for (uint32 i = 0; i < kFrameWidthPixels; ++i) {
    for (uint32 j = 0; j < kFrameHeightPixels; ++j) {
      colu_[offset] = Color::ABGRToAtariColor(image->GetPixel(i, j));
      ++offset;
    }
  }
}

void Frame::SetColor(uint32 offset, uint8 color) {
  assert(offset < kFrameSizeBytes);
  colu_[offset] = color;
}

void Frame::SetStrip(const std::unique_ptr<ColuStrip>& strip, uint32 row) {
  uint32 offset = row * kFrameWidthPixels;
  std::memcpy(colu_.get() + row, strip->colus(), kFrameWidthPixels);
}

std::unique_ptr<Image> Frame::ToImage() const {
  std::unique_ptr<Image> image(new Image(kFrameWidthPixels,
                                         kFrameHeightPixels));
  uint32 offset = 0;
  for (uint32 i = 0; i < kFrameWidthPixels; ++i) {
    for (uint32 j = 0; j < kFrameHeightPixels; ++j) {
      uint32 abgr = Color::AtariColorToABGR(colu_[offset]);
      image->SetPixel(i, j, abgr);
    }
  }

  return image;
}

std::unique_ptr<ColuStrip> Frame::GetStrip(uint32 row) const {
  assert(row < kFrameHeightPixels);
  return std::unique_ptr<ColuStrip>(
      new ColuStrip(colu_, row * kFrameWidthPixels));
}

}  // namespace vcsmc
