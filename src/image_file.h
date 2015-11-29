#ifndef SRC_IMAGE_FILE_H_
#define SRC_IMAGE_FILE_H_

#include <memory>
#include <string>

namespace vcsmc {

class Image;

// Class for loading and saving Image objects from supported image file formats,
// currently only PNG.
class ImageFile {
 public:
  static std::unique_ptr<Image> Load(const std::string& file_name);
  static bool Save(const Image* image, const std::string& file_name);

 private:
  static std::unique_ptr<Image> LoadPNG(const std::string& file_name);

  static bool SavePNG(const Image* image, const std::string& file_name);
};

}  // namespace vcsmc

#endif  // SRC_IMAGE_FILE_H_
