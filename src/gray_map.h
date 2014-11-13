#ifndef SRC_GRAY_MAP_H_
#define SRC_GRAY_MAP_H_

#include <memory>

#include "types.h"

namespace vcsmc {

// A GrayMap defines a field of floating-point values and has a width and
// height. It can be read from and saved to a grayscale PNG image file.
class GrayMap {
 public:
  GrayMap(uint32 width, uint32 height);

  // Loads a GrayMap from the provided monochromatic 8 or 16 bit .png file
  // and returns it, or nullptr on error.
  static std::unique_ptr<GrayMap> Load(const std::string& file_path);

  // Given a pointer to an 8-bit raw graymap saves it into a file. |bit_depth|
  // should be either 8 or 16.
  static void SaveFromBytes(const std::string& file_path, const uint8* bytes,
      uint32 width, uint32 height, int bit_depth);

  void Save(const std::string& file_path);

  uint32 width() const { return width_; }
  uint32 height() const { return height_; }
  const float* values() const { return values_.get(); }
  float* values_writeable() { return values_.get(); }

 private:
  uint32 width_;
  uint32 height_;
  std::unique_ptr<float[]> values_;
};

}  // namespace vcsmc

#endif  // SRC_GRAY_MAP_H_
