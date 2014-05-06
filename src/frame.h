#ifndef SRC_FRAME_H_
#define SRC_FRAME_H_

#include <memory>

#include "types.h"

namespace vcsmc {

class ColuStrip;
class Image;

// A Frame represents an Image mapped to Atari colors.
class Frame {
 public:
  // Build an empty black frame.
  Frame();

  // Build a Frame from an Image.
  Frame(Image* image);

  // Offset is (y * kFrameWidth) + x < kFrameSize;
  void SetColor(uint32 offset, uint8 color);

  // Set an entire strip at a time.
  void SetStrip(ColuStrip* strip, uint32 row);

  // Make a new Image from these frame colors and return.
  std::unique_ptr<Image> ToImage() const;

 private:
  std::unique_ptr<uint8[]> colu_;
};

}

#endif  // SRC_FRAME_H_
