#ifndef SRC_CL_IMAGE_IMPL_H_
#define SRC_CL_IMAGE_IMPL_H_

#include "cl_image.h"
#include "cl_include.h"
#include "types.h"

namespace vcsmc {

class Image;

class CLImageImpl : public CLImage {
 public:
  CLImageImpl(uint32 width, uint32 height);
  virtual ~CLImageImpl();

  bool Setup(cl_context context);

  virtual bool EnqueueCopyToDevice(CLCommandQueue* queue, Image* image)
      override;
  virtual bool EnqueueCopyFromDevice(CLCommandQueue* queue, Image* image)
      override;

  cl_mem get() const { return mem_; }

 private:
  cl_mem mem_;
  uint32 width_;
  uint32 height_;
};

}  // namespace vcsmc

#endif  // SRC_CL_IMAGE_IMPL_H_
