#ifndef SRC_TIFF_IMAGE_FILE_H_
#define SRC_TIFF_IMAGE_FILE_H_

#include "image_file.h"

class TiffImageFile : public ImageFile {
 public:
  TiffImageFile(const std::string& file_path) : ImageFile(file_path) {}

  // ImageFile overrides
  virtual Image* Load() override;
  virtual bool Save(Image* image) override;
};

#endif  // SRC_BITMAP_IMAGE_FILE_H_
