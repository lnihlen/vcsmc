#ifndef SRC_IMAGE_FILE_H_
#define SRC_IMAGE_FILE_H_

#include <memory>
#include <string>

#include "image.h"

namespace vcsmc {

// Abstract base class for defining image files, which can read and write Image
// objects.
class ImageFile {
 public:
  ImageFile(const std::string& file_path) : file_path_(file_path) {}

  virtual std::unique_ptr<Image> Load() = 0;
  virtual bool Save(Image* image) = 0;

 protected:
  std::string file_path_;

 private:
  // Default ctor private to disable.
  ImageFile() {}
};

}  // namespace vcsmc

#endif  // SRC_IMAGE_FILE_H_
