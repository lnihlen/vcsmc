#include <cmath>
#include <cstring>
#include <list>

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "cl_kernel.h"
#include "constants.h"
#include "gtest/gtest.h"

namespace vcsmc {

class FFTRadix2Test : public ::testing::Test {
 private:
  void QueueRadix2Round(
      CLCommandQueue* queue,
      std::list<std::unique_ptr<CLKernel>>& kernels,
      CLBuffer* first_buffer,
      CLBuffer* second_buffer,
      int n,
      int p,
      bool forward) {
    std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(
        CLProgram::Programs::kFFTRadix2));
    kernel->SetBufferArgument(0, forward ? first_buffer : second_buffer);
    kernel->SetByteArgument(1, sizeof(uint32), &p);
    kernel->SetBufferArgument(2, forward ? second_buffer : first_buffer);
    kernel->EnqueueWithGroupSize(queue, n / 2);
    queue->EnqueueBarrier();
    kernels.push_back(std::move(kernel));
  }

  void QueueInverseNorm(
      CLCommandQueue* queue,
      std::list<std::unique_ptr<CLKernel>>& kernels,
      CLBuffer* first_buffer,
      CLBuffer* second_buffer,
      float norm,
      bool forward) {
    std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(
        CLProgram::Programs::kInverseFFTNormalize));
    kernel->SetBufferArgument(0, forward ? first_buffer : second_buffer);
    kernel->SetByteArgument(1, sizeof(float), &norm);
    kernel->SetBufferArgument(2, forward ? second_buffer : first_buffer);
    kernel->Enqueue(queue);
    queue->EnqueueBarrier();
    kernels.push_back(std::move(kernel));
  }

 protected:
  // Performs the FFT. |input_data| points at n*2 floats, as does
  // |output_data|. If |forward| is true this performs a forward FFT, if false
  // it will perform an inverse FFT.
  void DoRadix2FFT(const float* input_data, uint32 n, bool forward,
      float* output_data) {
    ASSERT_TRUE((n != 0) && !(n & (n - 1)));
    std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
    std::unique_ptr<CLBuffer> first_buffer(
        CLDeviceContext::MakeBuffer(n * 2 * sizeof(float)));
    first_buffer->EnqueueCopyToDevice(queue.get(), input_data);
    std::unique_ptr<CLBuffer> second_buffer(
        CLDeviceContext::MakeBuffer(n * 2 * sizeof(float)));
    std::list<std::unique_ptr<CLKernel>> kernels;
    bool copying_forward = true;

    if (!forward) {
      QueueInverseNorm(queue.get(), kernels, first_buffer.get(),
          second_buffer.get(), 1.0f, copying_forward);
      copying_forward = !copying_forward;
    }

    uint32 p = 1;
    while (p < n) {
      QueueRadix2Round(queue.get(), kernels, first_buffer.get(),
          second_buffer.get(), n, p, copying_forward);
      copying_forward = !copying_forward;
      p = p << 1;
    }

    if (!forward) {
      QueueInverseNorm(queue.get(), kernels, first_buffer.get(),
          second_buffer.get(), (float)(n), copying_forward);
      copying_forward = !copying_forward;
    }

    // |copying_forward| means the data are going to be transfered from the
    // |first_buffer| to the |second_buffer| during processing. Therefore we
    // should copy the output data from the |first_buffer| if |copying_forward|
    // is true.
    if (copying_forward) {
      first_buffer->EnqueueCopyFromDevice(queue.get(), output_data);
    } else {
      second_buffer->EnqueueCopyFromDevice(queue.get(), output_data);
    }

    queue->Finish();
  }
};

// Test some fundamental FFT properties. A fundamental cos wave in the real
// values should result in an impulse in the real values at sample 1.
TEST_F(FFTRadix2Test, FFT2FundamentalCos) {
  const uint32 kN = 64;
  std::unique_ptr<float[]> input_data(new float[kN * 2]);
  for (uint32 i = 0; i < kN; ++i) {
    input_data[(i * 2) + 0] = cos(((double)i / (double)kN) * 2.0 * kPi);
    input_data[(i * 2) + 1] = 0.0f;
  }

  std::unique_ptr<float[]> output_data(new float[kN * 2]);
  DoRadix2FFT(input_data.get(), kN, true, output_data.get());
  EXPECT_NEAR(0.0f, output_data[0], 0.0001f);
  EXPECT_NEAR(0.0f, output_data[1], 0.0001f);
  EXPECT_NEAR((float)(kN / 2), output_data[2], 0.0001f);
  EXPECT_NEAR(0.0f, output_data[3], 0.0001f);

  for (uint32 i = 4; i < (kN * 2) - 2; ++i)
    EXPECT_NEAR(0.0f, output_data[i], 0.0001f);

  // Real-valued data has even symmetry in real transform values and odd
  // symmetry in imaginary transform values.
  EXPECT_NEAR((float)(kN / 2), output_data[(kN * 2) - 2], 0.0001f);
  EXPECT_NEAR(0.0f, output_data[(kN * 2) - 1], 0.0001f);

  // Inverse transform should match input signal, modulo noise.
  std::unique_ptr<float[]> inverse_data(new float[kN * 2]);
  DoRadix2FFT(output_data.get(), kN, false, inverse_data.get());
  for (uint32 i = 0; i < kN; ++i) {
    EXPECT_NEAR(input_data[(i * 2) + 0], inverse_data[(i * 2) + 0], 0.0001f);
    EXPECT_NEAR(input_data[(i * 2) + 1], inverse_data[(i * 2) + 1], 0.0001f);
  }
}

const float kRandomComplexFFT2TimeDomain[128 * 2] = {
  0.564340f, -0.295495f,
  0.076461f, -0.971177f,
  0.428296f, -0.192498f,
  0.591378f, -0.274385f,
  0.456864f, 0.372643f,
  -0.239011f, 0.648626f,
  -0.052654f, -0.504141f,
  -0.563763f, 0.919786f,
  -0.282140f, 0.964104f,
  0.904379f, 0.196518f,
  0.024364f, 0.735856f,
  0.783673f, 0.569357f,
  -0.277063f, -0.186329f,
  0.010073f, 0.263995f,
  -0.024117f, 0.144367f,
  0.733552f, -0.312819f,
  0.098340f, 0.463850f,
  0.325794f, -0.573518f,
  0.740623f, 0.183760f,
  -0.749882f, 0.814730f,
  -0.547065f, 0.116557f,
  0.869038f, -0.339264f,
  0.864241f, 0.399964f,
  -0.750487f, 0.971950f,
  -0.161910f, 0.742040f,
  -0.667814f, -0.438065f,
  0.638637f, -0.544263f,
  -0.441200f, -0.723866f,
  0.111097f, -0.171842f,
  -0.079087f, 0.888248f,
  0.631083f, -0.078607f,
  -0.014107f, -0.937673f,
  -0.661009f, 0.409576f,
  -0.982318f, -0.547399f,
  0.063115f, 0.341905f,
  0.639488f, -0.108361f,
  0.684174f, 0.798392f,
  0.919547f, -0.942122f,
  -0.721316f, 0.355748f,
  -0.098043f, 0.356837f,
  -0.828067f, 0.671689f,
  0.686225f, -0.705866f,
  -0.451069f, 0.516014f,
  0.331288f, -0.181151f,
  0.832675f, 0.907499f,
  0.467042f, 0.830054f,
  -0.448548f, 0.381390f,
  -0.653601f, -0.286959f,
  0.966950f, -0.418745f,
  0.950991f, 0.414679f,
  -0.069830f, -0.733295f,
  -0.014094f, 0.263038f,
  0.591816f, 0.490115f,
  -0.613816f, -0.291724f,
  0.675808f, 0.111627f,
  0.448957f, -0.865437f,
  -0.626494f, -0.142243f,
  -0.906061f, 0.878765f,
  0.297799f, -0.636484f,
  -0.142065f, 0.459934f,
  -0.765385f, 0.514188f,
  -0.492331f, 0.611254f,
  -0.769408f, -0.389674f,
  -0.558356f, -0.217197f,
  0.731111f, -0.982014f,
  -0.116725f, -0.950648f,
  -0.499375f, 0.808516f,
  0.787457f, -0.506709f,
  -0.771690f, -0.496046f,
  0.218735f, -0.343808f,
  -0.046020f, -0.849569f,
  0.909787f, 0.059302f,
  -0.016647f, 0.963209f,
  -0.063445f, -0.692626f,
  -0.032136f, -0.577478f,
  0.949825f, 0.654827f,
  -0.622512f, -0.498272f,
  0.066082f, 0.131915f,
  0.600058f, -0.556422f,
  0.516661f, 0.375001f,
  0.430957f, -0.293673f,
  0.915864f, 0.772083f,
  -0.188354f, 0.924699f,
  0.531782f, 0.512544f,
  0.783590f, -0.244597f,
  -0.213371f, 0.181187f,
  0.672511f, -0.404290f,
  0.420730f, -0.928204f,
  -0.624200f, 0.562350f,
  0.362676f, -0.242964f,
  -0.271133f, 0.341205f,
  0.168256f, 0.092871f,
  0.605368f, 0.923663f,
  0.977099f, 0.996647f,
  -0.290243f, -0.673715f,
  0.985840f, -0.972962f,
  0.684881f, -0.610044f,
  -0.671848f, 0.218892f,
  -0.366293f, 0.692200f,
  0.450306f, -0.816368f,
  0.189062f, 0.721365f,
  0.346670f, -0.944715f,
  0.181231f, 0.449354f,
  0.892574f, 0.737118f,
  0.690053f, 0.709765f,
  0.794501f, 0.871572f,
  -0.116805f, 0.879675f,
  -0.411717f, 0.748518f,
  0.256782f, 0.356871f,
  -0.335394f, 0.145071f,
  -0.330521f, -0.058614f,
  -0.629951f, 0.777398f,
  0.509386f, 0.834447f,
  0.585811f, -0.551734f,
  0.513080f, 0.845219f,
  0.048856f, -0.325672f,
  -0.216774f, 0.462565f,
  0.153844f, -0.631204f,
  -0.539170f, -0.877461f,
  0.552806f, -0.819634f,
  0.612909f, -0.954017f,
  -0.033895f, 0.396189f,
  -0.469821f, 0.724702f,
  -0.626282f, -0.012982f,
  -0.011927f, 0.638228f,
  -0.059985f, 0.969782f,
  -0.364427f, 0.449656f,
  0.652747f, -0.912796f
};

const float kRandomComplexFFT2FrequencyDomain[128 * 2] = {
  13.565219f, 8.897828f,
  0.611965f, 12.179090f,
  -6.302902f, -9.327662f,
  11.756291f, -2.922031f,
  0.300319f, -13.164149f,
  10.420291f, 3.974337f,
  -4.254568f, 5.967688f,
  0.596087f, 3.057540f,
  -4.853105f, -11.747066f,
  -2.456335f, -7.069277f,
  -0.279414f, -11.813186f,
  -1.252598f, 4.527982f,
  -9.503467f, -4.002417f,
  -1.100981f, -6.806020f,
  1.851508f, -9.147634f,
  3.868222f, -14.179171f,
  6.660125f, -7.613698f,
  -0.755320f, 13.597890f,
  12.795482f, -6.998806f,
  -7.398189f, -7.805843f,
  -4.607749f, 1.167275f,
  -2.312217f, 8.263560f,
  -5.834496f, 4.896775f,
  -6.183546f, 7.679364f,
  9.249334f, -13.743182f,
  -9.506924f, 2.176037f,
  -2.924916f, -8.679245f,
  -3.374316f, -1.241400f,
  6.279403f, 2.628724f,
  10.486632f, 5.264063f,
  -2.362369f, -8.222968f,
  0.776315f, -11.042285f,
  4.246473f, 6.707140f,
  -2.513543f, 2.274384f,
  2.533483f, 11.649022f,
  11.355144f, -5.152407f,
  8.205417f, 9.878214f,
  3.336966f, 6.099369f,
  0.529595f, -11.093342f,
  -8.000245f, 12.240917f,
  9.501105f, 2.647379f,
  3.956421f, 11.521828f,
  -5.092790f, -5.859369f,
  8.369184f, -6.409211f,
  2.679430f, 0.628764f,
  -2.444579f, -0.621003f,
  -4.021620f, -1.847725f,
  -7.620131f, 6.844632f,
  -6.320966f, 3.146147f,
  -0.933734f, -0.042654f,
  8.383410f, 1.073268f,
  -8.706319f, -4.371667f,
  -2.668345f, 6.244335f,
  2.511981f, 0.326776f,
  -1.829924f, 4.240507f,
  0.433841f, 10.034538f,
  -4.765625f, 0.377012f,
  1.607106f, 4.503730f,
  3.288107f, 5.819992f,
  12.478408f, -0.280619f,
  -4.106445f, -4.471820f,
  -2.205937f, 2.110730f,
  4.033078f, -10.612690f,
  0.936728f, 9.330881f,
  -6.231068f, 10.180466f,
  5.341908f, 0.314146f,
  -6.675040f, -4.615057f,
  -7.176525f, 0.375822f,
  2.517219f, 15.578899f,
  4.591219f, -12.486395f,
  1.941012f, -2.534879f,
  -3.999920f, 2.296223f,
  13.258786f, -0.657183f,
  -1.557376f, 8.917530f,
  -6.746773f, 9.449457f,
  -12.703636f, -5.579085f,
  3.103844f, -0.737566f,
  -4.531161f, -2.884401f,
  -6.096537f, 6.791867f,
  0.483946f, -6.007289f,
  7.739006f, 3.048116f,
  3.476366f, -0.336831f,
  13.069691f, -0.925866f,
  -6.953386f, 9.670372f,
  10.044614f, -11.883431f,
  0.520515f, 4.954079f,
  9.927403f, -4.906971f,
  -13.160983f, -7.996846f,
  3.119771f, -0.064855f,
  -4.736760f, 5.366068f,
  -2.260094f, 3.189237f,
  -3.180805f, -4.846100f,
  -4.134518f, 4.652445f,
  -1.096105f, -1.435650f,
  2.452805f, -9.084440f,
  -3.711464f, -0.008782f,
  1.969262f, 3.533771f,
  6.330813f, 5.313940f,
  -2.455205f, 0.640764f,
  0.828406f, 2.357075f,
  -4.354269f, -17.195568f,
  -4.341501f, 2.799005f,
  14.274412f, 4.370236f,
  5.691979f, 2.441918f,
  -0.713026f, -12.042882f,
  -4.069339f, -3.909100f,
  0.510403f, 4.616376f,
  0.173835f, -5.603349f,
  -12.697185f, -8.614659f,
  -3.035392f, -4.304137f,
  6.150088f, 1.536680f,
  1.698644f, 3.943394f,
  -4.920375f, -6.901369f,
  4.786264f, 0.711141f,
  6.988066f, 2.251571f,
  -7.613293f, -5.709058f,
  9.877435f, -10.039565f,
  16.598022f, 3.643381f,
  10.590766f, -1.360856f,
  3.739131f, -5.210061f,
  11.694373f, -0.041193f,
  -0.483669f, -1.664347f,
  1.585343f, 7.449466f,
  -2.252414f, -1.722681f,
  -11.916638f, -3.691174f,
  0.795577f, 2.285818f,
  -3.077493f, -1.377539f,
  2.137133f, 0.187356f
};

// Compare FFT on random complex data to values calculated by numpy.
TEST_F(FFTRadix2Test, FFT2RandomComplexData) {
  const uint32 kN = 128;
  std::unique_ptr<float[]> output_data(new float[kN * 2]);
  // Check forward transform first.
  DoRadix2FFT(kRandomComplexFFT2TimeDomain, kN, true, output_data.get());
  for (uint32 i = 0; i < kN; ++i) {
    EXPECT_NEAR(kRandomComplexFFT2FrequencyDomain[(i * 2) + 0],
        output_data[(i * 2) + 0], 0.0001f);
    EXPECT_NEAR(kRandomComplexFFT2FrequencyDomain[(i * 2) + 1],
        output_data[(i * 2) + 1], 0.0001f);
  }

  // Check inverse transform.
  DoRadix2FFT(kRandomComplexFFT2FrequencyDomain, kN, false, output_data.get());
  for (uint32 i = 0; i < kN; ++i) {
    printf("%d\n", i);
    EXPECT_NEAR(kRandomComplexFFT2TimeDomain[(i * 2) + 0],
        output_data[(i * 2) + 0], 0.0001f);
    EXPECT_NEAR(kRandomComplexFFT2TimeDomain[(i * 2) + 1],
        output_data[(i * 2) + 1], 0.0001f);
  }
}

}  // namespace vcsmc
