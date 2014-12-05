#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_kernel.h"
#include "gtest/gtest.h"

namespace vcsmc {

TEST(DownsampleErrorsTest, MultiRowConstantWeighting) {
  const uint32 kInputWidth = 25;
  const uint32 kOutputWidth = 6;
  std::unique_ptr<float[]> input_floats(new float[kInputWidth * 3]);
  for (uint32 i = 0; i < kInputWidth; ++i) {
    input_floats[i] = 1.0f;
    input_floats[i + kInputWidth] = 2.0f;
    input_floats[i + (2 * kInputWidth)] = 3.0f;
  }

  std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
  std::unique_ptr<CLBuffer> input_buffer(CLDeviceContext::MakeBuffer(
      kInputWidth * 3 * sizeof(float)));
  input_buffer->EnqueueCopyToDevice(queue.get(), input_floats.get());
  std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(
      CLProgram::kDownsampleErrors));
  kernel->SetBufferArgument(0, input_buffer.get());
  uint32 input_width = kInputWidth;
  kernel->SetByteArgument(1, sizeof(uint32), &input_width);
  std::unique_ptr<CLBuffer> output_buffer(CLDeviceContext::MakeBuffer(
      kOutputWidth * 3 * sizeof(float)));
  kernel->SetBufferArgument(2, output_buffer.get());
  kernel->Enqueue2D(queue.get(), kOutputWidth, 3);

  std::unique_ptr<float[]> output_floats(new float[kOutputWidth * 3]);
  output_buffer->EnqueueCopyFromDevice(queue.get(), output_floats.get());
  queue->Finish();

  for (uint32 i = 0; i < kOutputWidth; ++i) {
    EXPECT_NEAR((float)kInputWidth / (float)kOutputWidth, output_floats[i],
        0.001f);
    EXPECT_NEAR(2.0f * (float)kInputWidth / (float)kOutputWidth,
        output_floats[i + kOutputWidth], 0.001f);
    EXPECT_NEAR(3.0f * (float)kInputWidth / (float)kOutputWidth,
        output_floats[i + (2 * kOutputWidth)], 0.001f);
  }
}

TEST(DownsampleErrorsTest, DoubleRowIncreasingWeighting) {
  const uint32 kInputWidth = 32;
  const uint32 kOutputWidth = 8;
  std::unique_ptr<float[]> input_floats(new float[kInputWidth * 2]);
  for (uint32 i = 0; i < kInputWidth; ++i) {
    input_floats[i] = 1.0f;
    input_floats[i + kInputWidth] = (float)(i * kOutputWidth / kInputWidth);
  }

  std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
  std::unique_ptr<CLBuffer> input_buffer(CLDeviceContext::MakeBuffer(
      kInputWidth * 2 * sizeof(float)));
  input_buffer->EnqueueCopyToDevice(queue.get(), input_floats.get());
  std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(
      CLProgram::kDownsampleErrors));
  kernel->SetBufferArgument(0, input_buffer.get());
  uint32 input_width = kInputWidth;
  kernel->SetByteArgument(1, sizeof(uint32), &input_width);
  std::unique_ptr<CLBuffer> output_buffer(CLDeviceContext::MakeBuffer(
      kOutputWidth * 2 * sizeof(float)));
  kernel->SetBufferArgument(2, output_buffer.get());
  kernel->Enqueue2D(queue.get(), kOutputWidth, 2);

  std::unique_ptr<float[]> output_floats(new float[kOutputWidth * 2]);
  output_buffer->EnqueueCopyFromDevice(queue.get(), output_floats.get());
  queue->Finish();

  for (uint32 i = 0; i < kOutputWidth; ++i) {
    EXPECT_NEAR((float)kInputWidth / (float)kOutputWidth, output_floats[i],
        0.001f);
    EXPECT_NEAR((float)kInputWidth / (float)kOutputWidth * (float)i,
        output_floats[i + kOutputWidth], 0.001f);
  }
}

}  // namespace vcsmc
