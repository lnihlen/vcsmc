#ifndef SRC_CL_IMAGE_IMPL_H_
#define SRC_CL_IMAGE_IMPL_H_

#include "cl_image.h"
#include "cl_include.h"
#include "types.h"

namespace vcsmc {

class Image;
class PixelStrip;

class CLImageImpl : public CLImage {
 public:
  CLImageImpl(const Image* image);
  CLImageImpl(const PixelStrip* strip);
  ~CLImageImpl();

  bool Setup(cl_context context);

  virtual bool EnqueueCopyToDevice(CLCommandQueue* queue) override;

  const cl_mem get() const { return mem_; }

 private:
  cl_mem mem_;
  uint32 width_;
  uint32 height_;
  // non-owning pointer!
  const uint32* pixels_;
};

}  // namespace vcsmc

#endif  // SRC_CL_IMAGE_IMPL_H_
