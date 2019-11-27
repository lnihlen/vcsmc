#include "image_file.h"

#include "atari_ntsc_rgb_color_table.h"
#include "constants.h"

#include <cassert>
#include <cstring>
#include <png.h>
#include <stdio.h>

namespace vcsmc {

bool LoadImage(const std::string& file_name, uint8* planes) {
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

  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
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
  png_get_IHDR(png_ptr, info_ptr, &width, &height, (int*)&bit_depth, &color_type, nullptr, nullptr, nullptr);
  if (bit_depth != 8 || (color_type != PNG_COLOR_TYPE_RGB && color_type != PNG_COLOR_TYPE_RGB_ALPHA)) {
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

bool SaveImage(const uint8* planes, size_t width, size_t height, const std::string& file_name) {
  // Code almost entirely copied from writepng.c by Greg Roelofs.
  FILE* png_file = fopen(file_name.c_str(), "wb");
  if (!png_file)
    return false;

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
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
               PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png_ptr, info_ptr);
  png_set_packing(png_ptr);

  std::unique_ptr<uint8[]> pixels(new uint8[width * height * 3]);
  const uint8* red_plane = planes;
  const uint8* green_plane = planes + (width * height);
  const uint8* blue_plane = planes + (2 * width * height);
  for (size_t i = 0; i < width * height; ++i) {
    pixels.get()[(i * 3)] = red_plane[i];
    pixels.get()[(i * 3) + 1] = green_plane[i];
    pixels.get()[(i * 3) + 2] = blue_plane[i];
  }

  png_bytepp row_pointers = (png_bytepp)malloc(height * sizeof(png_bytep));
  for (size_t i = 0; i < height; ++i) {
    row_pointers[i] = pixels.get() + (i * width * 3);
  }

  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, NULL);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  free(row_pointers);
  fclose(png_file);
  return true;
}

bool SaveAtariPaletteImage(const uint8* indices, const std::string& fileName) {
    FILE* pngFile = fopen(fileName.c_str(), "rb");
    if (!pngFile) return false;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    assert(png);

    png_infop info = png_create_info_struct(png);
    assert(info);

    png_init_io(png, pngFile);
    png_set_compression_level(png, 6);
    png_set_IHDR(png, info, kTargetFrameWidthPixels, kFrameHeightPixels, 8, PNG_COLOR_TYPE_PALETTE, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // Pack the planar atari colors into a RGB color table.
    png_colorp palette = (png_colorp)png_malloc(png, 128 * sizeof(png_color));
    for (auto i = 0; i < 128; ++i) {
        palette[i].red = kAtariNtscRedColorTable[i];
        palette[i].green = kAtariNtscGreenColorTable[i];
        palette[i].blue = kAtariNtscBlueColorTable[i];
    }

    png_set_PLTE(png, info, palette, 128);
    png_write_info(png, info);
    png_set_packing(png);
    png_bytepp rowPointers = (png_bytepp)malloc(kFrameHeightPixels * sizeof(png_bytep));
    // To get around the const of the input pointer we make a temporary copy here of the image index bytes. Yes I am
    // aware this is horrible.
    std::unique_ptr<uint8[]> indexCopy(new uint8[kFrameSizeBytes]);
    std::memcpy(indexCopy.get(), indices, kFrameSizeBytes);
    for (auto i = 0; i < kFrameHeightPixels; ++i) {
        rowPointers[i] = indexCopy.get() + (i * kFrameWidthPixels);
    }

    png_write_image(png, rowPointers);
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(pngFile);
    free(rowPointers);
    free(palette);
    return true;
}

}  // namespace vcsmc

