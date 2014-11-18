#ifndef SRC_CL_IMAGE_H_
#define SRC_CL_IMAGE_H_

namespace vcsmc {

class CLCommandQueue;
class Image;

class CLImage {
 public:
  virtual bool EnqueueCopyToDevice(CLCommandQueue* queue) = 0;

  virtual ~CLImage() {}
};

}  // namespace vcsmc

#endif  // SRC_CL_IMAGE_H_
