#include "image_file.h"

#include <cassert>
#include <png.h>
#include <stdio.h>

namespace {

bool LoadPNG(const std::string& file_name, uint8* planes) {
  // Code almost entirely copied from readpng.c by Greg Roelofs.
  FILE* png_file = fopen(file_name.c_str(), "rb");
  if (!png_file)
    return false;

  uint8 png_sig[8];
  fread(png_sig, 1, 8, png_file);
  if (!png_check_sig(png_sig, 8)) {
    fclose(png_file);
    return false;
  }

  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
      nullptr, nullptr, nullptr);
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
      &color_type, nullptr, nullptr, nullptr);
  if (bit_depth != 8 ||
      (color_type != PNG_COLOR_TYPE_RGB &&
          color_type != PNG_COLOR_TYPE_RGB_ALPHA)) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    fclose(png_file);
    return false;
  }

  // Ask PNG to conform to ARGB (RGBA little-endian) color layout.
  if (color_type == PNG_COLOR_TYPE_RGB)
    png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);
  png_read_update_info(png_ptr, info_ptr);

  png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
  uint32 bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);

  std::unique_ptr<uint8[]> interleaved(new uint8[width * height * 4]);
  for (uint32 i = 0; i < height; ++i) {
    row_pointers[i] = interleaved.get() + (i * bytes_per_row);
  }
  png_read_image(png_ptr, row_pointers);
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(png_file);

  uint8* red_plane = planes;
  uint8* green_plane = planes + (width * height);
  uint8* blue_plane = planes + (2 * width * height);
  for (uint32 i = 0; i < (width * height); ++i) {
    red_plane[i] = interleaved.get()[(i * 4)];
    green_plane[i] = interleaved.get()[(i * 4) + 1];
    blue_plane[i] = interleaved.get()[(i * 4) + 2];
  }

  return true;
}

bool SavePNG(const uint8* planes, size_t width, size_t height,
             const std::string& file_name) {
  // Code almost entirely copied from writepng.c by Greg Roelofs.
  FILE* png_file = fopen(file_name.c_str(), "wb");
  if (!png_file)
    return false;

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
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_ptr, info_ptr);
  png_set_packing(png_ptr);

  std::unique_ptr<uint8[]> pixels(new uint8[width * height * 4]);
  const uint8* red_plane = planes;
  const uint8* green_plane = planes + (width * height);
  const uint8* blue_plane = planes + (2 * width * height);
  for (size_t i = 0; i < width * height; ++i) {
    pixels.get()[(i * 4)] = red_plane[i];
    pixels.get()[(i * 4) + 1] = green_plane[i];
    pixels.get()[(i * 4) + 2] = blue_plane[i];
    pixels.get()[(i * 4) + 3] = 0xff;
  }

  png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
  for (size_t i = 0; i < height; ++i) {
    row_pointers[i] = pixels.get() + (i * width * 4);
  }

  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, NULL);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  free(row_pointers);
  fclose(png_file);
  return true;
}

}

namespace vcsmc {

bool LoadImage(const std::string& file_name, uint8* planes) {
  size_t ext_pos = file_name.find_last_of(".");
  if (ext_pos == std::string::npos)
    return false;

  std::string ext = file_name.substr(ext_pos);
  if (ext == ".png")
    return LoadPNG(file_name, planes);

  return false;
}

bool SaveImage(const uint8* planes, size_t width, size_t height,
               const std::string& file_name) {
  size_t ext_pos = file_name.find_last_of(".");
  if (ext_pos == std::string::npos)
    return false;

  std::string ext = file_name.substr(ext_pos);
  if (ext == ".png")
    return SavePNG(planes, width, height, file_name);

  return false;
}

}  // namespace vcsmc
