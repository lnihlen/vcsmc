#include "pallette.h"

#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "pixel_strip.h"
#include "random.h"

namespace vcsmc {

Pallette::Pallette(uint32 num_colus)
    : num_colus_(num_colus) {
}

// For now we compute based on histogram only. Ultimately if the PixelStrip
// has an weight strip we may want to use that to guide our color choices.
void Pallette::Compute(
    PixelStrip* pixel_strip, CLCommandQueue* queue, Random* random) {
  // Pick num_colus_ colors at random.
  vector<uint8> colous;
  colus.reserve(num_colus_);
  for (size_t i = 0; i < num_culos_; ++i)
    colus.push_back(random->next() % 128);

  // Make num_colus_ buffers for calculating errors.
  vector<std::unique_ptr<CLBuffer>> buffers;
  buffers.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i) {
    buffers.push_back(
        std::move(CLDeviceContext::MakeBuffer(128 * sizeof(float))));
  }

  vector<std::unique_ptr<CLKernel>> kernels;
  kernels.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i) {
    std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(kCiede2k));
    kernel->SetBufferArgument(0, pixel_strip->lab_strip());
    kernel->SetBufferArgument(1, Color::AtariLabColorBuffer(colus[i]);
    kernel->SetBufferArgument(2, buffers[i]);
    kernel->Enqueue(queue);
    kernels.push_back(std::move(kernel));
  }

  // Make num_colus_ arrays to hold resulting error vectors.
  vector<std::unique_ptr<float[]>> results;
  // and so on and so forth.
}

}  // namespace vcsmc
