#include "value_map.h"

#include <cassert>
#include <png.h>
#include <stdio.h>
#include <string>

namespace vcsmc {

ValueMap::ValueMap(uint32 width, uint32 height)
    : width_(width),
      height_(height) {
}

// static
void ValueMap::SaveFromBytes(const std::string& file_path, const uint8* bytes,
    uint32 width, uint32 height, uint32 bit_depth, uint32 bytes_per_row) {
  // Code almost entirely copied from writepng.c by Greg Roelofs.
  FILE* png_file = fopen(file_path.c_str(), "wb");
  if (!png_file)
    return;

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL,
      NULL, NULL);
  assert(png_ptr);

  png_infop info_ptr = png_create_info_struct(png_ptr);
  assert(info_ptr);

  png_init_io(png_ptr, png_file);
  png_set_compression_level(png_ptr, 6);
  png_set_IHDR(png_ptr,
               info_ptr,
               width,
               height,
               bit_depth,
               PNG_COLOR_TYPE_GRAY,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_ptr, info_ptr);
  png_set_packing(png_ptr);

  png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
  for (uint32 i = 0; i < height; ++i)
    row_pointers[i] = const_cast<png_bytep>(bytes + (i * bytes_per_row));

  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, NULL);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  free(row_pointers);
  fclose(png_file);
}

// static
std::unique_ptr<uint8[]> ValueMap::LoadFromFile(const std::string& file_path,
    uint32& width_out, uint32& height_out, uint32& bit_depth_out,
    uint32& bytes_per_row_out) {
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
  uint32 bit_depth = 0;
  int color_type;
  png_get_IHDR(png_ptr, info_ptr, &width, &height, (int*)&bit_depth,
      &color_type, NULL, NULL, NULL);
  if ((bit_depth != 16 && bit_depth != 8 && bit_depth != 1) ||
        color_type != PNG_COLOR_TYPE_GRAY) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(png_file);
    return nullptr;
  }

  // No transformations to register, so just update info.
  png_read_update_info(png_ptr, info_ptr);

  png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
  uint32 bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);

  std::unique_ptr<uint8[]> image_bytes(new uint8[bytes_per_row * height]);
  for (uint32 i = 0; i < height; ++i)
    row_pointers[i] = image_bytes.get() + (i * bytes_per_row);
  png_read_image(png_ptr, row_pointers);
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(png_file);

  width_out = width;
  height_out = height;
  bit_depth_out = bit_depth;
  bytes_per_row_out = bytes_per_row;
  return std::move(image_bytes);
}

}  // namespace vcsmc
