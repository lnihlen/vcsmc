#include "tiff_image_file.h"

#include <cstring>
#include <tiffio.h>

#include "types.h"

namespace vcsmc {

std::unique_ptr<Image> TiffImageFile::Load() {
  TIFF* tiff = TIFFOpen(file_path_.c_str(), "r");
  if (!tiff)
    return nullptr;

  uint32 width, height;
  uint32* pixels;
  TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);

  std::unique_ptr<Image> image(new Image(width, height));

  if (!TIFFReadRGBAImage(tiff, width, height, image->pixels_writeable(), 0))
    return nullptr;

  TIFFClose(tiff);

  return image;
}

bool TiffImageFile::Save(const std::unique_ptr<Image>& image) {
  // Open file for writing.
  TIFF* tiff = TIFFOpen(file_path_.c_str(), "w");
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
  uint8* scanline = NULL;
  uint32 bytes_per_line = 4 * image->width();
  uint32 scanline_size = TIFFScanlineSize(tiff);
  if (bytes_per_line < scanline_size) {
    scanline = new uint8[scanline_size];
    // clear upper unused bytes to zero
    std::memset(scanline + bytes_per_line, 0, scanline_size - bytes_per_line);
  }

  TIFFSetField(tiff,
               TIFFTAG_ROWSPERSTRIP,
               TIFFDefaultStripSize(tiff, bytes_per_line));

  // Write out the image line-by-line.
  bool success = true;
  uint32* line = image->pixels_writeable();
  if (scanline) {
    for (uint32 i = 0; i < image->height(); ++i) {
      std::memcpy(scanline, line, bytes_per_line);
      if (TIFFWriteScanline(tiff, scanline, i, 0) < 0) {
        success = false;
        break;
      }
      line += image->width();
    }
  } else {
    for (uint32 i = 0; i < image->height(); ++i) {
      if (TIFFWriteScanline(tiff, line, i, 0) < 0) {
        success = false;
        break;
      }
      line += image->width();
    }
  }

  TIFFClose(tiff);
  delete[] scanline;
  return success;
}

}  // namespace vcsmc
