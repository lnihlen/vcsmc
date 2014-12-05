#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_kernel.h"
#include "gtest/gtest.h"
#include "types.h"

namespace vcsmc {

TEST(KMeansTest, Classify) {
  const uint32 kPixelCount = 32;
  std::unique_ptr<float[]> color_errors(new float[kPixelCount * 3]);
  uint32 colors[3] = { 0, 1, 2 };
  for (uint32 i = 0; i < kPixelCount; ++i) {
    color_errors[i] = (i % 3) ? 1.0f : 0.0f;
    color_errors[i + kPixelCount] = ((i + 2) % 3) ? 1.0f : 0.0f;
    color_errors[i + (kPixelCount * 2)] = ((i + 1) % 3) ? 1.0f : 0.0f;
  }

  std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
  std::unique_ptr<CLBuffer> errors_buffer(
      CLDeviceContext::MakeBuffer(kPixelCount * 3 * sizeof(float)));
  errors_buffer->EnqueueCopyToDevice(queue.get(), color_errors.get());
  std::unique_ptr<CLBuffer> colors_buffer(
      CLDeviceContext::MakeBuffer(3 * sizeof(uint32)));
  colors_buffer->EnqueueCopyToDevice(queue.get(), colors);

  std::unique_ptr<CLKernel> classify_kernel(
      CLDeviceContext::MakeKernel(CLProgram::Programs::kKMeansClassify));
  classify_kernel->SetBufferArgument(0, errors_buffer.get());
  classify_kernel->SetBufferArgument(1, colors_buffer.get());
  uint32 num_classes = 3;
  classify_kernel->SetByteArgument(2, sizeof(uint32), &num_classes);
  std::unique_ptr<CLBuffer> classes_buffer(
      CLDeviceContext::MakeBuffer(kPixelCount * sizeof(uint32)));
  classify_kernel->SetBufferArgument(3, classes_buffer.get());
  classify_kernel->Enqueue(queue.get(), kPixelCount);

  std::unique_ptr<uint32[]> classes(new uint32[kPixelCount]);
  classes_buffer->EnqueueCopyFromDevice(queue.get(), classes.get());

  queue->Finish();

  for (uint32 i = 0; i < kPixelCount; ++i)
    EXPECT_EQ(i % 3, classes[i]);
}

TEST(KMeansTest, Color) {
  const uint32 kPixelCount = 27;
  std::unique_ptr<float[]> color_errors(new float[kPixelCount * 4]);
  std::unique_ptr<uint32[]> classes(new uint32[kPixelCount]);
  for (uint32 i = 0; i < kPixelCount; ++i) {
    color_errors[i] = (i % 4) ? 1.0f : 0.1f;
    color_errors[i + kPixelCount] = ((i + 3) % 4) ? 1.0f : 0.1f;
    color_errors[i + (kPixelCount * 2)] = ((i + 2) % 4) ? 1.0f : 0.1f;
    color_errors[i + (kPixelCount * 3)] = ((i + 1) % 4) ? 1.0f : 0.1f;
    classes[i] = i % 4;
  }

  std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
  std::unique_ptr<CLBuffer> errors_buffer(
      CLDeviceContext::MakeBuffer(kPixelCount * 4 * sizeof(float)));
  errors_buffer->EnqueueCopyToDevice(queue.get(), color_errors.get());
  std::unique_ptr<CLBuffer> classes_buffer(
      CLDeviceContext::MakeBuffer(kPixelCount * sizeof(uint32)));
  classes_buffer->EnqueueCopyToDevice(queue.get(), classes.get());

  std::unique_ptr<CLKernel> color_kernel(
      CLDeviceContext::MakeKernel(CLProgram::Programs::kKMeansColor));
  color_kernel->SetBufferArgument(0, errors_buffer.get());
  color_kernel->SetBufferArgument(1, classes_buffer.get());
  uint32 image_width = kPixelCount;
  color_kernel->SetByteArgument(2, sizeof(uint32), &image_width);
  uint32 num_classes = 4;
  color_kernel->SetByteArgument(3, sizeof(uint32), &num_classes);
  uint32 iteration = 0;
  color_kernel->SetByteArgument(4, sizeof(uint32), &iteration);
  color_kernel->SetByteArgument(5, sizeof(float) * 4 * 4, nullptr);
  color_kernel->SetByteArgument(6, sizeof(uint32) * 4 * 4, nullptr);
  std::unique_ptr<CLBuffer> fit_error_buffer(
      CLDeviceContext::MakeBuffer(sizeof(float)));
  color_kernel->SetBufferArgument(7, fit_error_buffer.get());
  std::unique_ptr<CLBuffer> colors_buffer(
      CLDeviceContext::MakeBuffer(4 * sizeof(uint32)));
  color_kernel->SetBufferArgument(8, colors_buffer.get());
  color_kernel->Enqueue(queue.get(), 4);

  float fit_error = 0.0f;
  fit_error_buffer->EnqueueCopyFromDevice(queue.get(), &fit_error);
  std::unique_ptr<uint32[]> colors(new uint32[4]);
  colors_buffer->EnqueueCopyFromDevice(queue.get(), colors.get());
  queue->Finish();

  for (uint32 i = 0; i < 4; ++i)
    EXPECT_EQ(i, colors[i]);

  EXPECT_NEAR((float)kPixelCount * 0.1f, fit_error, 0.001f);
}

}  // namespace vcsmc
