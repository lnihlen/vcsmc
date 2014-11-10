#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "cl_kernel.h"
#include "constants.h"
#include "gtest/gtest.h"

namespace vcsmc {

class InverseFFTNormTest : public ::testing::Test {
 protected:
  // Applies the inverse_fft_normalize shader to the n float2s pointed at by
  // |input_data| with the |scalar| and copies the results to |output_data|.
  void DoInverseNorm(const float* input_data, uint32 n, float scalar,
        float* output_data) {
    std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
    std::unique_ptr<CLBuffer> in_buffer = CLDeviceContext::MakeBuffer(
        n * 2 * sizeof(float));
    ASSERT_TRUE(in_buffer->EnqueueCopyToDevice(queue.get(), input_data));
    std::unique_ptr<CLBuffer> out_buffer = CLDeviceContext::MakeBuffer(
        n * 2 * sizeof(float));
    std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(
        CLProgram::Programs::kInverseFFTNormalize));
    ASSERT_TRUE(kernel->SetBufferArgument(0, in_buffer.get()));
    ASSERT_TRUE(kernel->SetByteArgument(1, sizeof(float), &scalar));
    ASSERT_TRUE(kernel->SetBufferArgument(2, out_buffer.get()));
    ASSERT_TRUE(kernel->Enqueue(queue.get(), n / 2));
    ASSERT_TRUE(out_buffer->EnqueueCopyFromDevice(queue.get(), output_data));
    queue->Finish();
  }
};

TEST_F(InverseFFTNormTest, ConjugateWithoutScale) {
  const uint32 kN = 32;
  std::unique_ptr<float[]> input_data(new float[kN * 2]);
  for (uint32 i = 0; i < kN; ++i) {
    input_data[(i * 2) + 0] = (float)(i + kN);
    input_data[(i * 2) + 1] = (float)(kN - i) * ((i % 2) ? 1.0f : -1.0f);
  }
  std::unique_ptr<float[]> output_data(new float[kN * 2]);
  DoInverseNorm(input_data.get(), kN, 1.0f, output_data.get());
  for (uint32 i = 0; i < kN; ++i) {
    EXPECT_EQ(input_data[(i * 2) + 0], output_data[(i * 2) + 0]);
    EXPECT_EQ(-1.0f * input_data[(i * 2) + 1], output_data[(i * 2) + 1]);
  }
}

TEST_F(InverseFFTNormTest, ConjugateWithScale) {
  const uint32 kN = 32;
  std::unique_ptr<float[]> input_data(new float[kN * 2]);
  for (uint32 i = 0; i < kN; ++i) {
    input_data[(i * 2) + 0] = (float)(i + kN);
    input_data[(i * 2) + 1] = (float)(kN - i) * ((i % 2) ? 1.0f : -1.0f);
  }
  std::unique_ptr<float[]> output_data(new float[kN * 2]);
  DoInverseNorm(input_data.get(), kN, (float)kN, output_data.get());
  for (uint32 i = 0; i < kN; ++i) {
    EXPECT_EQ(input_data[(i * 2) + 0] / (float)kN, output_data[(i * 2) + 0]);
    EXPECT_EQ(-1.0f * input_data[(i * 2) + 1] / (float)kN,
        output_data[(i * 2) + 1]);
  }
}


}  // namespace vcsmc
