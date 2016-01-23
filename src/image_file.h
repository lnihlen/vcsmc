#ifndef SRC_IMAGE_FILE_H_
#define SRC_IMAGE_FILE_H_

#include <memory>
#include <string>

namespace vcsmc {

class Image;

std::unique_ptr<Image> LoadImage(const std::string& file_name);
bool SaveImage(const Image* image, const std::string& file_name);

}  // namespace vcsmc

#endif  // SRC_IMAGE_FILE_H_
