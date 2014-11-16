#ifndef SRC_BIT_MAP_H_
#define SRC_BIT_MAP_H_

#include "value_map.h"

namespace vcsmc {

// A BitMap defines a field of single-bit values and has a width and a height.
// It can be read from and saved to a bitmap PNG image file. Designed to store
// the bitmap as memory efficiently as possible.
class BitMap : public ValueMap {
 public:
  BitMap(uint32 width, uint32 height);

  // Loads a BitMap from the provided monochromatic 1-bit .png file and returns
  // it, or nullptr on error.
  static std::unique_ptr<BitMap> Load(const std::string& file_path);

  void Save(const std::string& file_path);

  // Given a pointer to bytemap with |bytes_per_row| >= |width_| BitMap will
  // build its internal bitwise representation to match.
  void Pack(const uint8* bytes, uint32 bytes_per_row_unpacked);

  // Sets the bit at |x, y| to |value|.
  void SetBit(uint32 x, uint32 y, bool value);

  // Returns the value of the bitmap at (x, y).
  bool bit(uint32 x, uint32 y);

 private:
  BitMap(uint32 width, uint32 height, std::unique_ptr<uint8[]> bytes,
      uint32 bytes_per_row);
  uint32 bytes_per_row_;
  std::unique_ptr<uint8[]> bytes_;
};

}  // namespace vcsmc

#endif  // SRC_BIT_MAP_H_
