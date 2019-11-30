#ifndef SRC_IMAGE_FILE_H_
#define SRC_IMAGE_FILE_H_

#include <memory>
#include <string>

#include "types.h"

namespace vcsmc {

// currently only supporting PNGs
bool LoadImage(const std::string& file_name, uint8* planes);
bool SaveImage(const uint8* planes, size_t width, size_t height,
               const std::string& file_name);

bool SaveAtariPaletteImage(const uint8* indices, const std::string& fileName);

}  // namespace vcsmc

#endif  // SRC_IMAGE_FILE_H_
