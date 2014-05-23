#include "cl_image_impl.h"

#include <OpenCL/OpenCL.h>

#include "cl_command_queue_impl.h"
#include "image.h"
#include "pixel_strip.h"

namespace vcsmc {

CLImageImpl::CLImageImpl(const Image* image)
    : mem_(0),
      width_(image->width()),
      height_(image->height()),
      pixels_(image->pixels()) {
}

CLImageImpl::CLImageImpl(const PixelStrip* strip)
    : mem_(0),
      width_(strip->width()),
      height_(1),
      pixels_(strip->pixels()) {
}

CLImageImpl::~CLImageImpl() {
  clReleaseMemObject(mem_);
}

bool CLImageImpl::Setup(cl_context context) {
  cl_image_format image_format;
  image_format.image_channel_order = CL_RGBA;
  image_format.image_channel_data_type = CL_UNORM_INT8;

  cl_image_desc image_desc;
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  image_desc.image_width = width_;
  image_desc.image_height = height_;
  image_desc.image_depth = 1;
  image_desc.image_array_size = 1;
  image_desc.image_row_pitch = width_ * 4;
  image_desc.num_mip_levels = 0;
  image_desc.num_samples = 0;
  image_desc.buffer = NULL;

  int result = 0;
  mem_ = clCreateImage(context,
                       CL_MEM_READ_ONLY,
                       &image_format,
                       &image_desc,
                       NULL,
                       &result);

  return mem_ && result == CL_SUCCESS;
}

bool CLImageImpl::EnqueueCopyToDevice(CLCommandQueue* queue) {
  CLCommandQueueImpl* command_queue = static_cast<CLCommandQueueImpl*>(queue);
  size_t origin[3] = { 0, 0, 0 };
  size_t region[3] = { width_, height_, 1 };
  int result = clEnqueueWriteImage(command_queue->get(),
                                   mem_,
                                   false,
                                   origin,
                                   region,
                                   width_ * 4,
                                   1,
                                   pixels_,
                                   0,
                                   NULL,
                                   NULL);
  return result == CL_SUCCESS;
}

}  // namespace vcsmc
