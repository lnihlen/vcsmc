#include "image_file.h"

#include <cassert>
#include <png.h>
#include <stdio.h>
#include <string>

#include "image.h"
#include "cl_image.h"
#include "types.h"

namespace vcsmc {

// static
std::unique_ptr<Image> ImageFile::Load(const std::string& file_name) {
  size_t ext_pos = file_name.find_last_of(".");
  if (ext_pos == std::string::npos)
    return nullptr;

  std::string ext = file_name.substr(ext_pos);
  if (ext == ".png")
    return LoadPNG(file_name);

  return nullptr;
}

// static
bool ImageFile::Save(const Image* image, const std::string& file_name) {
  size_t ext_pos = file_name.find_last_of(".");
  if (ext_pos == std::string::npos)
    return false;

  std::string ext = file_name.substr(ext_pos);
  if (ext == ".png")
    return SavePNG(image, file_name);

  return false;
}

// static
std::unique_ptr<Image> ImageFile::LoadPNG(const std::string& file_name) {
  // Code almost entirely copied from readpng.c by Greg Roelofs.
  FILE* png_file = fopen(file_name.c_str(), "rb");
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
  if (bit_depth != 8 ||
      (color_type != PNG_COLOR_TYPE_RGB &&
          color_type != PNG_COLOR_TYPE_RGB_ALPHA)) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(png_file);
    return nullptr;
  }

  // Ask PNG to conform to ARGB (RGBA little-endian) color layout.
  if (color_type == PNG_COLOR_TYPE_RGB)
    png_set_filler(png_ptr, 0xff, PNG_FILLER_AFTER);
  png_read_update_info(png_ptr, info_ptr);

  png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
  uint32 bytes_per_row = png_get_rowbytes(png_ptr, info_ptr);

  std::unique_ptr<Image> image(new Image(width, height));
  for (uint32 i = 0; i < height; ++i) {
    row_pointers[i] = reinterpret_cast<uint8*>(image->pixels_writeable())
         + (i * bytes_per_row);
  }
  png_read_image(png_ptr, row_pointers);
  free(row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(png_file);

  return std::move(image);
}

// static
bool ImageFile::SavePNG(const Image* image, const std::string& file_name) {
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
               image->width(),
               image->height(),
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_ptr, info_ptr);
  png_set_packing(png_ptr);

  png_bytepp row_pointers = (png_bytepp)malloc(
      image->height() * sizeof(png_bytep));
  for (uint32 i = 0; i < image->height(); ++i) {
    row_pointers[i] =
        reinterpret_cast<uint8*>(const_cast<uint32*>(image->pixels()))
            + (i * image->width() * sizeof(uint32));
  }

  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, NULL);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  free(row_pointers);
  fclose(png_file);
  return true;
}

}  // namespace vcsmc
