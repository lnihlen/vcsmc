#include "image.h"

namespace vcsmc {

Image::Image(uint32 width, uint32 height)
    : width_(width),
      height_(height),
      pixels_(new uint32[width * height]) {
}

Image::~Image() {
}

}  // namespace vcsmc
