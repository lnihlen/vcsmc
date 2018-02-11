#include "gray_map.h"

namespace vcsmc {

GrayMap::GrayMap(uint32 width, uint32 height)
    : ValueMap(width, height),
      values_(new float[width * height]) {
}

GrayMap::~GrayMap() {
}

// static
std::unique_ptr<GrayMap> GrayMap::Load(const std::string& file_path) {
  // Load png image bytes.
  uint32 width, height, bit_depth, bytes_per_row;
  std::unique_ptr<uint8[]> image_bytes(ValueMap::LoadFromFile(file_path,
      width, height, bit_depth, bytes_per_row));
  if (!image_bytes || bit_depth < 8)
    return nullptr;

  // Build GrayMap from image.
  std::unique_ptr<GrayMap> gray_map(new GrayMap(width, height));
  if (bit_depth == 8) {
    float* gray_value = gray_map->values_writeable();
    for (uint32 i = 0; i < height; ++i) {
      uint8* gray_byte = image_bytes.get() + (i * bytes_per_row);
      for (uint32 j = 0; j < width; ++j) {
        *gray_value = static_cast<float>(*gray_byte) / 255.0;
        ++gray_byte;
        ++gray_value;
      }
    }
  } else {
    float* gray_value = gray_map->values_writeable();
    for (uint32 i = 0; i < height; ++i) {
      uint16* gray_short = reinterpret_cast<uint16*>(
          image_bytes.get() + (i * bytes_per_row * 2));
      for (uint32 j = 0; j < width; ++j) {
        *gray_value = static_cast<float>(*gray_short) / 65535.0;
        ++gray_short;
        ++gray_value;
      }
    }
  }
  return gray_map;
}

// static
void GrayMap::Save(const float* map,
                   uint32 width,
                   uint32 height,
                   const std::string& file_path) {
  // Convert from floats to 16-bit grays and then save.
  std::unique_ptr<uint16[]> gray_shorts(new uint16[width * height]);
  const float* val_ptr = map;
  uint16* short_ptr = gray_shorts.get();
  for (uint32 i = 0; i < width * height; ++i) {
    *short_ptr = static_cast<uint16>(*val_ptr * 65535.0);
    ++short_ptr;
    ++val_ptr;
  }
  ValueMap::SaveFromBytes(file_path, reinterpret_cast<const uint8*>(
      gray_shorts.get()), width, height, 16, width * 2);
}

void GrayMap::Save(const std::string& file_path) {
  Save(values_.get(), width_, height_, file_path);
}

}  // namespace vcsmc
