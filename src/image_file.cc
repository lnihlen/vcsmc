#include "image_file.h"

#include <cassert>
#include <png.h>
#include <stdio.h>
#include <string>
#include <tiffio.h>

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
  else if (ext == ".tiff")
    return LoadTIFF(file_name);

  return nullptr;
}

// static
bool ImageFile::Save(const Image* image, const std::string& file_name) {
  size_t ext_pos = file_name.find_last_of(".");
  if (ext_pos == std::string::npos)
    return nullptr;

  std::string ext = file_name.substr(ext_pos);
  if (ext == ".png")
    return SavePNG(image, file_name);
  else if (ext == ".tiff")
    return SaveTIFF(image, file_name);

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
std::unique_ptr<Image> ImageFile::LoadTIFF(const std::string& file_name) {
  TIFF* tiff = TIFFOpen(file_name.c_str(), "r");
  if (!tiff)
    return nullptr;

  uint32 width, height;
  TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);

  std::unique_ptr<Image> image(new Image(width, height));

  if (!TIFFReadRGBAImage(tiff, width, height, image->pixels_writeable(), 0))
    return nullptr;

  TIFFClose(tiff);

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

// static
bool ImageFile::SaveTIFF(const Image* image, const std::string& file_name) {
  // Open file for writing.
  TIFF* tiff = TIFFOpen(file_name.c_str(), "w");
  if (!tiff) return false;

  // Set metadata
  TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, image->width());
  TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, image->height());
  TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 4);
  TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

  // TIFF scanlines may be wider than the width of the Image.
  std::unique_ptr<uint8[]> scanline;
  uint32 bytes_per_line = 4 * image->width();
  uint32 scanline_size = TIFFScanlineSize(tiff);
  if (bytes_per_line < scanline_size) {
    scanline.reset(new uint8[scanline_size]);
    // clear upper unused bytes to zero
    std::memset(scanline.get() + bytes_per_line, 0,
        scanline_size - bytes_per_line);
  }

  TIFFSetField(tiff,
               TIFFTAG_ROWSPERSTRIP,
               TIFFDefaultStripSize(tiff, bytes_per_line));

  // Write out the image line-by-line.
  const uint32* line = image->pixels();
  if (scanline) {
    for (uint32 i = 0; i < image->height(); ++i) {
      std::memcpy(scanline.get(), line, bytes_per_line);
      if (TIFFWriteScanline(tiff, scanline.get(), i, 0) < 0)
        return false;
      line += image->width();
    }
  } else {
    for (uint32 i = 0; i < image->height(); ++i) {
      if (TIFFWriteScanline(tiff, const_cast<uint32*>(line), i, 0) < 0)
        return false;
      line += image->width();
    }
  }

  TIFFClose(tiff);
  return true;
}

}  // namespace vcsmc
