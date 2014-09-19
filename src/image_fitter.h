#ifndef SRC_IMAGE_FITTER_H_
#define SRC_IMAGE_FITTER_H_

#include <memory>
#include <vector>

#include "types.h"

namespace vcsmc {

class Image;
class Spec;

class ImageFitter {
 public:
  ImageFitter(std::unique_ptr<Image> image);
  std::unique_ptr<std::vector<Spec>> Fit(uint64 base_frame_time);

 private:

  std::unique_ptr<Image> image_;
};

}  // namespace vcsmc

#endif  // SRC_IMAGE_FITTER_H_
