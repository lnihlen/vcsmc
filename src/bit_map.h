#ifndef SRC_BIT_MAP_H_
#define SRC_BIT_MAP_H_

#include <memory>

#include "types.h"

namespace vcsmc {

// A BitMap defines a field of single-bit values and has a width and a height.
// It can be read from and saved to a bitmap PNG image file. Designed to store
// the bitmap as memory efficiently as possible.
class BitMap {
 public:
  BitMap(uint32 width, uint32 height);

  // Loads a BitMap from the provided monochromatic 1-bit .png file and returns
  // it, or nullptr on error.
  static std::unique_ptr<BitMap> Load(const std::string& file_path);

  void Save(const std::string& file_path);

  // Returns the value of the bitmap at (x, y).
  bool bit(uint32 x, uint32 y);

  uint32 width() const { return width_; }
  uint32 height() const { return height_; }

 private:
  uint32 width_;
  uint32 byte_width_;
  uint32 height_;
  std::unique_ptr<uint8[]> bytes_;
};

}  // namespace vcsmc

#endif  // SRC_BIT_MAP_H_
