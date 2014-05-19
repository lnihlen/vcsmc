#ifndef SRC_CL_IMAGE_IMPL_H_
#define SRC_CL_IMAGE_IMPL_H_

#include <OpenCL/OpenCL.h>

#include "cl_image.h"

namespace vcsmc {

class Image;

class CLImageImpl : public CLImage {
 public:
  CLImageImpl(const Image* image);
  ~CLImageImpl();

  bool Setup(cl_context context);

  virtual bool EnqueueCopyToDevice(CLCommandQueue* queue) override;

 private:
  cl_mem mem_;
  // non-owning pointer
  const Image* image_;
};

}  // namespace vcsmc

#endif  // SRC_CL_IMAGE_IMPL_H_
