#ifndef SRC_GRAY_MAP_H_
#define SRC_GRAY_MAP_H_

#include "value_map.h"

namespace vcsmc {

// A GrayMap defines a field of floating-point values and has a width and
// height. It can be read from and saved to a grayscale PNG image file.
class GrayMap : public ValueMap {
 public:
  GrayMap(uint32 width, uint32 height);
  virtual ~GrayMap();

  // Loads a GrayMap from the provided monochromatic 8 or 16 bit .png file
  // and returns it, or nullptr on error.
  static std::unique_ptr<GrayMap> Load(const std::string& file_path);

  // Saves a GrayMap given a float array, dimensions, and a file_path.
  static void Save(const float* map,
                   uint32 width,
                   uint32 height,
                   const std::string& file_path);

  void Save(const std::string& file_path);

  const float* values() const { return values_.get(); }
  float* values_writeable() { return values_.get(); }

 private:
  std::unique_ptr<float> values_;
};

}  // namespace vcsmc

#endif  // SRC_GRAY_MAP_H_
