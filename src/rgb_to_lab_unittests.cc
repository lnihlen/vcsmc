#include <cstring>

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "cl_kernel.h"
#include "color_table.h"
#include "gtest/gtest.h"
#include "image.h"

namespace vcsmc {

// We use a python colormath script to preconvert the 128 Atari NTSC colors in
// to Lab colorspace. This test validates our OpenCL conversion against the
// colormath one.
TEST(RgbToLabTest, TestAtariColorConversion) {
  // Build an Image object 128 pixels wide and 1 pixel tall to contain the Atari
  // color table data, convert it and test the results against our constants.
  std::unique_ptr<Image> atari_colors_rgb_image(new Image(128, 1));
  std::memcpy(atari_colors_rgb_image->pixels_writeable(),
      kAtariNTSCABGRColorTable, 128 * 4);
  std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
  atari_colors_rgb_image->CopyToDevice(queue.get());
  std::unique_ptr<CLBuffer> lab_results_buffer(
      CLDeviceContext::MakeBuffer(128 * 4 * sizeof(float)));
  std::unique_ptr<CLKernel> kernel(
      CLDeviceContext::MakeKernel(CLProgram::Programs::kRGBToLab));
  kernel->SetImageArgument(0, atari_colors_rgb_image->cl_image());
  int first_row = 0;
  kernel->SetByteArgument(1, sizeof(int), &first_row);
  kernel->SetBufferArgument(2, lab_results_buffer.get());
  kernel->Enqueue(queue.get(), 128);
  std::unique_ptr<float[]> lab_results(new float[128 * 4]);
  lab_results_buffer->EnqueueCopyFromDevice(queue.get(), lab_results.get());
  queue->Finish();

  for (int i = 0; i < 128; ++i) {
    EXPECT_NEAR(
        kAtariNTSCLabColorTable[(i * 4) + 0], lab_results[(i * 4) + 0], 0.01);
    EXPECT_NEAR(
        kAtariNTSCLabColorTable[(i * 4) + 1], lab_results[(i * 4) + 1], 0.01);
    EXPECT_NEAR(
        kAtariNTSCLabColorTable[(i * 4) + 2], lab_results[(i * 4) + 2], 0.01);
  }
}

}  // namespace vcsmc
