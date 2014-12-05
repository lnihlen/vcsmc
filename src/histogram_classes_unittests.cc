#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_kernel.h"
#include "gtest/gtest.h"
#include "types.h"

namespace vcsmc {

TEST(HistogramClassesTest, HistogramClasses) {
  const uint32 kPixelCount = 48;
  std::unique_ptr<uint32[]> classes(new uint32[kPixelCount]);
  for (uint32 i = 0; i < kPixelCount; ++i)
    classes[i] = i % 7;

  std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
  std::unique_ptr<CLBuffer> classes_buffer(
      CLDeviceContext::MakeBuffer(kPixelCount * sizeof(uint32)));
  classes_buffer->EnqueueCopyToDevice(queue.get(), classes.get());

  std::unique_ptr<CLKernel> histo_kernel(
      CLDeviceContext::MakeKernel(CLProgram::Programs::kHistogramClasses));
  histo_kernel->SetBufferArgument(0, classes_buffer.get());
  uint32 num_classes = 7;
  histo_kernel->SetByteArgument(1, sizeof(uint32), &num_classes);
  histo_kernel->SetByteArgument(2, sizeof(uint32) * 7 * kPixelCount, nullptr);
  std::unique_ptr<CLBuffer> counts_buffer(CLDeviceContext::MakeBuffer(
      sizeof(uint32) * 7));
  histo_kernel->SetBufferArgument(3, counts_buffer.get());
  histo_kernel->Enqueue(queue.get(), kPixelCount);
  std::unique_ptr<uint32[]> counts(new uint32[7]);
  counts_buffer->EnqueueCopyFromDevice(queue.get(), counts.get());
  queue->Finish();

  EXPECT_EQ(7, counts[0]);
  EXPECT_EQ(7, counts[1]);
  EXPECT_EQ(7, counts[2]);
  EXPECT_EQ(7, counts[3]);
  EXPECT_EQ(7, counts[4]);
  EXPECT_EQ(7, counts[5]);
  EXPECT_EQ(6, counts[6]);
}

}  // namespace vcsmc
