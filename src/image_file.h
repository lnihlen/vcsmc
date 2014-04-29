#ifndef SRC_IMAGE_FILE_H_
#define SRC_IMAGE_FILE_H_

#include <string>

#include "image.h"

// Abstract base class for defining image files, which can read and write Image
// objects.
class ImageFile {
 public:
  ImageFile(const std::string& file_path) : file_path_(file_path) {}

  virtual Image* Load() = 0;
  virtual bool Save(Image* image) = 0;

 protected:
  std::string file_path_;

 private:
  // Default ctor private to disable.
  ImageFile() {}
};


#endif  // SRC_IMAGE_FILE_H_
