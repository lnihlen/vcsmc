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

TEST(LabToRGBTest, TestAtariColorConversion) {
  std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
  std::unique_ptr<CLBuffer> lab_input_buffer(
      CLDeviceContext::MakeBuffer(128 * 4 * sizeof(float)));
  lab_input_buffer->EnqueueCopyToDevice(queue.get(), kAtariNTSCLabColorTable);
  std::unique_ptr<CLImage> rgb_image(CLDeviceContext::MakeImage(128, 1));
  ASSERT_TRUE(rgb_image.get());
  std::unique_ptr<CLKernel> kernel(
      CLDeviceContext::MakeKernel(CLProgram::Programs::kLabToRGB));
  ASSERT_TRUE(kernel->SetBufferArgument(0, lab_input_buffer.get()));
  ASSERT_TRUE(kernel->SetImageArgument(1, rgb_image.get()));
  ASSERT_TRUE(kernel->Enqueue2D(queue.get(), 128, 1));

  Image result_image(128, 1);
  rgb_image->EnqueueCopyFromDevice(queue.get(), &result_image);
  queue->Finish();

  const uint8* table = reinterpret_cast<const uint8*>(kAtariNTSCABGRColorTable);
  const uint8* results = reinterpret_cast<const uint8*>(result_image.pixels());
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(table[(i * 4) + 0], results[(i * 4) + 0]);
    EXPECT_EQ(table[(i * 4) + 1], results[(i * 4) + 1]);
    EXPECT_EQ(table[(i * 4) + 2], results[(i * 4) + 2]);
    EXPECT_EQ(table[(i * 4) + 3], results[(i * 4) + 3]);
  }
}

}  // namespace vcsmc
