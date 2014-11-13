#include "bit_map.h"

#include <cassert>
#include <png.h>
#include <stdio.h>
#include <string>

namespace vcsmc {

BitMap::BitMap(uint32 width, uint32 height)
    : width_(width),
      byte_width_(width_ / 8),
      height_(height) {
  if (width_ % 8)
    ++byte_width_;
  bytes_.reset(new uint8[byte_width_ * height_]);
}

// static
std::unique_ptr<BitMap> BitMap::Load(const std::string& file_path) {
/*
  // Code almost entirely copied from readpng.c by Greg Roelofs.
  FILE* png_file = fopen(file_path.c_str(), "rb");
  if (!png_file)
    return nullptr;

  uint8 png_sig[8];
  fread(png_sig, 1, 8, png_file);
  if (!png_check_sig(png_sig, 8)) {
    fclose(png_file);
    return nullptr;
  }

  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
      NULL, NULL, NULL);
  assert(png_ptr);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  assert(info_ptr);

  png_init_io(png_ptr, png_file);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);

  uint32 width = 0;
  uint32 height = 0;
  int bit_depth, color_type;
  png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
      NULL, NULL, NULL);
  if (bit_depth != 1 || color_type != PNG_COLOR_TYPE_GRAY) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(png_file);
    return nullptr;
  }

  // No transformations to register, so just update info.
  png_read_update_info(png_ptr, info_ptr);

  png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
  uint32 bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);

  // question - is this coming in packed or unpacked.

  std::unique_ptr<uint8[]> image_bytes(new uint8[bytes_per_row * height]);
  for (uint32 i = 0; i < height; ++i)
    row_pointers[i] = image_bytes.get() + (i * bytes_per_row);
  png_read_image(png_ptr, row_pointers);

  // Build bitmap, if needed.

  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(png_file);
  return std::move(gray_map);
*/
  return nullptr;
}

void BitMap::Save(const std::string& file_path) {
}

bool BitMap::bit(uint32 x, uint32 y) {
  return false;
}

}  // namespace vcsmc
