#include "frame.h"

#include <cassert>
#include <cstring>

#include "color.h"

Frame::Frame() {
  colors_ = new uint8[kFrameSize];
  memset(colors_, 0, kFrameSize);
}

Frame::Frame(Image* image) {
  assert(image->width() == kFrameWidth);
  assert(image->height() == kFrameHeight);

  colors_ = new uint8[kFrameSize];
  uint32 offset = 0;
  for (uint32 i = 0; i < kFrameWidth; ++i) {
    for (uint32 j = 0; j < kFrameHeight; ++j) {
      colors_[offset] = Color::ABGRToAtariColor(image->GetPixel(i, j));
      ++offset;
    }
  }
}

Frame::Frame(const Frame& frame) {
  colors_ = new uint8[kFrameSize];
  *this = frame;
}

const Frame& Frame::operator=(const Frame& frame) {
  memcpy(colors_, frame.colors_, kFrameSize);
  return *this;
}

Frame::~Frame() {
  delete[] colors_;
}

void Frame::SetColor(uint32 offset, uint8 color) {
  assert(offset < kFrameSize);
  colors_[offset] = color;
}

Image* Frame::ToImage() {
  Image* image = new Image(kFrameWidth, kFrameHeight);
  uint32 offset = 0;
  for (uint32 i = 0; i < kFrameWidth; ++i) {
    for (uint32 j = 0; j < kFrameHeight; ++j) {
      uint32 abgr = Color::AtariColorToABGR(colors_[offset]);
      image->SetPixel(i, j, abgr);
    }
  }

  return image;
}
