#include "gray_map.h"

#include <cassert>
#include <png.h>
#include <stdio.h>
#include <string>

namespace vcsmc {

GrayMap::GrayMap(uint32 width, uint32 height)
    : width_(width),
      height_(height),
      values_(new float[width * height]) {
}


// static
std::unique_ptr<GrayMap> GrayMap::Load(const std::string& file_path) {
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
  if ((bit_depth != 16 && bit_depth != 8) ||
        color_type != PNG_COLOR_TYPE_GRAY) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(png_file);
    return nullptr;
  }

  // No transformations to register, so just update info.
  png_read_update_info(png_ptr, info_ptr);

  png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
  uint32 bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);
  if (bit_depth == 8) {
    assert(bytes_per_row == width);
  } else {
    assert(bit_depth == 16);
    assert(bytes_per_row = width * 2);
  }

  std::unique_ptr<uint8[]> image_bytes(new uint8[bytes_per_row * height]);
  for (uint32 i = 0; i < height; ++i)
    row_pointers[i] = image_bytes.get() + (i * bytes_per_row);
  png_read_image(png_ptr, row_pointers);

  // Build GrayMap from image.
  std::unique_ptr<GrayMap> gray_map(new GrayMap(width, height));
  if (bit_depth == 8) {
    uint8* gray_byte = image_bytes.get();
    float* gray_value = gray_map->values_writeable();
    for (int i = 0; i < width * height; ++i) {
      *gray_value = static_cast<float>(*gray_byte) / 255.0f;
      ++gray_byte;
      ++gray_value;
    }
  } else {
    uint16* gray_short = reinterpret_cast<uint16*>(image_bytes.get());
    float* gray_value = gray_map->values_writeable();
    for (int i = 0; i < width * height; ++i) {
      *gray_value = static_cast<float>(*gray_short) / 65535.0f;
      ++gray_short;
      ++gray_value;
    }
  }
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(png_file);
  return std::move(gray_map);
}

// static
void GrayMap::SaveFromBytes(const std::string& file_path, const uint8* bytes,
    uint32 width, uint32 height, int bit_depth) {
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
  if (bit_depth == 8) {
    for (uint32 i = 0; i < height; ++i)
      row_pointers[i] = const_cast<png_bytep>(bytes + (i * width));
  } else {
    assert(bit_depth == 16);
    for (uint32 i = 0; i < height; ++i)
      row_pointers[i] = const_cast<png_bytep>(bytes + (i * width * 2));
  }

  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, NULL);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  free(row_pointers);
  fclose(png_file);
}

void GrayMap::Save(const std::string& file_path) {
  // Convert from floats to 16-bit grays and then save.
  std::unique_ptr<uint16[]> gray_shorts(new uint16[width_ * height_]);
  float* val_ptr = values_.get();
  uint16* short_ptr = gray_shorts.get();
  for (uint32 i = 0; i < width_ * height_; ++i) {
    *short_ptr = static_cast<uint16>(*val_ptr * 65535.0f);
    ++short_ptr;
    ++val_ptr;
  }
  SaveFromBytes(file_path, reinterpret_cast<const uint8*>(gray_shorts.get()),
      width_, height_, 16);
}

}  // namespace vcsmc
