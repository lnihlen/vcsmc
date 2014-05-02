#ifndef SRC_TIFF_IMAGE_FILE_H_
#define SRC_TIFF_IMAGE_FILE_H_

#include "image_file.h"

namespace vcsmc {

class TiffImageFile : public ImageFile {
 public:
  TiffImageFile(const std::string& file_path) : ImageFile(file_path) {}

  // ImageFile overrides
  virtual std::unique_ptr<Image> Load() override;
  virtual bool Save(const std::unique_ptr<Image>& image) override;
};

}  // namespace vcsmc

#endif  // SRC_BITMAP_IMAGE_FILE_H_
