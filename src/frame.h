#ifndef SRC_FRAME_H_
#define SRC_FRAME_H_

#include "image.h"
#include "types.h"

const uint32 kFrameWidth = 160;
const uint32 kFrameHeight = 192;
const uint32 kFrameSize = kFrameWidth * kFrameHeight;

// A Frame represents an Image mapped to Atari colors.
class Frame {
 public:
  // Build an empty black frame.
  Frame();
  Frame(const Frame& frame);
  const Frame& operator=(const Frame& frame);
  // Build a Frame from an Image.
  Frame(Image* image);
  ~Frame();

  // Offset is (y * kFrameWidth) + x < kFrameSize;
  void SetColor(uint32 offset, uint8 color);

  // Make a new Image from these frame colors and return.
  Image* ToImage();

 private:
  uint8* colors_;
};

#endif  // SRC_FRAME_H_
