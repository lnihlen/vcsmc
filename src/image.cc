#include "image.h"

#include <cassert>
#include <cstring>

#include "cl_command_queue.h"
#include "cl_buffer.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "cl_kernel.h"
#include "color.h"
#include "constants.h"

namespace vcsmc {

Image::Image(uint32 width, uint32 height)
    : width_(width),
      height_(height),
      pixels_(new uint32[width * height]) {
}

Image::~Image() {
}

bool Image::CopyToDevice(CLCommandQueue* queue) {
  cl_image_ = CLDeviceContext::MakeImage(this);
  if (!cl_image_)
    return false;
  return cl_image_->EnqueueCopyToDevice(queue);
}

}  // namespace vcsmc
