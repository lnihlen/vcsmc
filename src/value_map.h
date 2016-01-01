#ifndef SRC_VALUE_MAP_H_
#define SRC_VALUE_MAP_H_

#include <string>

#include "types.h"

namespace vcsmc {

class ValueMap {
 public:
  ValueMap(uint32 width, uint32 height);

  uint32 width() const { return width_; }
  uint32 height() const { return height_; }

  // Given a pointer to an 8-bit raw graymap saves it into a file. |bit_depth|
  // should 16, 8, or 1.
  static void SaveFromBytes(const std::string& file_path, const uint8* bytes,
      uint32 width, uint32 height, uint32 bit_depth, uint32 bytes_per_row);

 protected:
  // Attempts to load the .png file at |file_path|. If successful will return
  // a valid pointer to the packed image bytes along with populating width,
  // height, and bit depth arguments.
  static std::unique_ptr<uint8[]> LoadFromFile(const std::string& file_path,
      uint32& width_out, uint32& height_out, uint32& bit_depth_out,
      uint32& bytes_per_row_out);

  uint32 width_;
  uint32 height_;
};

}  // namespace vcsmc

#endif  // SRC_VALUE_MAP_H_
