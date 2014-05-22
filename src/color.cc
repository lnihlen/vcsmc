#include "color.h"

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "constants.h"

// include generated color table file
#include "auto/color_table.cc"

namespace vcsmc {

Color* Color::instance_ = NULL;

// static
uint32 Color::AtariColorToABGR(uint8 atari_color) {
  return kAtariNTSCABGRColorTable[atari_color / 2];
}

// static
const float* Color::AtariColorToLab(uint8 atari_color) {
  return kAtariNTSCLabColorTable + (atari_color * 2);
}

// static
const CLBuffer* Color::AtariLabColorBuffer(uint8 atari_color) {
  return instance_->atari_color_buffers_[atari_color / 2].get();
}

// static
bool Color::Setup() {
  instance_ = new Color;
  return instance_->CopyColorBuffers();
}

// static
void Color::Teardown() {
  delete instance_;
  instance_ = NULL;
}

bool Color::CopyColorBuffers() {
  // Make an array of 128 floats of correct width, for copying to card.
  std::vector<std::unique_ptr<float[]>> color_strips;
  color_strips.reserve(128);
  for (size_t i = 0; i < 128; ++i) {
    std::unique_ptr<float[]> color_strip(new float[4 * kFrameWidthPixels]);
    for (size_t j = 0; j < 4 * kFrameWidthPixels; ++j)
      color_strip[j] = kAtariNTSCLabColorTable[(i * 4) + (j % 4)];
    color_strips.push_back(std::move(color_strip));
  }

  // Make 128 buffers on OpenCL side to store the data
  atari_color_buffers_.reserve(128);
  for (size_t i = 0; i < 128; ++i) {
    std::unique_ptr<CLBuffer> buffer = CLDeviceContext::MakeBuffer(
        sizeof(float) * 4 * kFrameWidthPixels);
    if (!buffer)
      return false;
    atari_color_buffers_.push_back(std::move(buffer));
  }

  // Enqueue 128 transfers to OpenCL device
  std::unique_ptr<CLCommandQueue> queue(CLDeviceContext::MakeCommandQueue());
  if (!queue)
    return false;

  for (size_t i = 0; i < 128; ++i) {
    atari_color_buffers_[i]->EnqueueCopyToDevice(
        queue.get(), color_strips[i].get());
  }

  // Wait for completion.
  queue->Finish();

  return true;
}

}  // namespace vcsmc
