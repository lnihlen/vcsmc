#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_kernel.h"
#include "gtest/gtest.h"

namespace vcsmc {

// Test example data taken from the helpful implementation notes and advice in:
// http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
// "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary
// Test Data, and Mathematical Observations", G. Sharma, W. Wu, E. N. Dalal,
// Color Research and Application, vol. 30. No. 1, February 2005
TEST(Ciede2kTest, SharmaWuBalalTestData) {
  const int kSWBTestDataCount = 34;

  const float lab_1[kSWBTestDataCount * 4] = {
    50.0000,   2.6772, -79.7751,   0.0000,
    50.0000,   3.1571, -77.2803,   0.0000,
    50.0000,   2.8361, -74.0200,   0.0000,
    50.0000,  -1.3802, -84.2814,   0.0000,
    50.0000,  -1.1848, -84.8006,   0.0000,
    50.0000,  -0.9009, -85.5211,   0.0000,
    50.0000,   0.0000,   0.0000,   0.0000,
    50.0000,  -1.0000,   2.0000,   0.0000,
    50.0000,   2.4900,  -0.0010,   0.0000,
    50.0000,   2.4900,  -0.0010,   0.0000,
    50.0000,   2.4900,  -0.0010,   0.0000,
    50.0000,   2.4900,  -0.0010,   0.0000,
    50.0000,  -0.0010,   2.4900,   0.0000,
    50.0000,  -0.0010,   2.4900,   0.0000,
    50.0000,  -0.0010,   2.4900,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    50.0000,   2.5000,   0.0000,   0.0000,
    60.2574, -34.0099,  36.2677,   0.0000,
    63.0109, -31.0961,  -5.8663,   0.0000,
    61.2901,   3.7196,  -5.3901,   0.0000,
    35.0831, -44.1164,   3.7933,   0.0000,
    22.7233,  20.0904, -46.6940,   0.0000,
    36.4612,  47.8580,  18.3852,   0.0000,
    90.8027,  -2.0831,   1.4410,   0.0000,
    90.9257,  -0.5406,  -0.9208,   0.0000,
     6.7747,  -0.2908,  -2.4247,   0.0000,
     2.0776,   0.0795,  -1.1350,   0.0000
  };

  const float lab_2[kSWBTestDataCount * 4] = {
    50.0000,   0.0000, -82.7485,   0.0000,
    50.0000,   0.0000, -82.7485,   0.0000,
    50.0000,   0.0000, -82.7485,   0.0000,
    50.0000,   0.0000, -82.7485,   0.0000,
    50.0000,   0.0000, -82.7485,   0.0000,
    50.0000,   0.0000, -82.7485,   0.0000,
    50.0000,  -1.0000,   2.0000,   0.0000,
    50.0000,   0.0000,   0.0000,   0.0000,
    50.0000,  -2.4900,   0.0009,   0.0000,
    50.0000,  -2.4900,   0.0010,   0.0000,
    50.0000,  -2.4900,   0.0011,   0.0000,
    50.0000,  -2.4900,   0.0012,   0.0000,
    50.0000,   0.0009,  -2.4900,   0.0000,
    50.0000,   0.0010,  -2.4900,   0.0000,
    50.0000,   0.0011,  -2.4900,   0.0000,
    50.0000,   0.0000,  -2.5000,   0.0000,
    73.0000,  25.0000, -18.0000,   0.0000,
    61.0000,  -5.0000,  29.0000,   0.0000,
    56.0000, -27.0000,  -3.0000,   0.0000,
    58.0000,  24.0000,  15.0000,   0.0000,
    50.0000,   3.1736,   0.5854,   0.0000,
    50.0000,   3.2972,   0.0000,   0.0000,
    50.0000,   1.8634,   0.5757,   0.0000,
    50.0000,   3.2592,   0.3350,   0.0000,
    60.4626, -34.1751,  39.4387,   0.0000,
    62.8187, -29.7946,  -4.0864,   0.0000,
    61.4292,   2.2480,  -4.9620,   0.0000,
    35.0232, -40.0716,   1.5901,   0.0000,
    23.0331,  14.9730, -42.5619,   0.0000,
    36.2715,  50.5065,  21.2231,   0.0000,
    91.1528,  -1.6435,   0.0447,   0.0000,
    88.6381,  -0.8985,  -0.7239,   0.0000,
     5.8714,  -0.0985,  -2.2286,   0.0000,
     0.9033,  -0.0636,  -0.5514,   0.0000
  };

  const float expected_results[kSWBTestDataCount] = {
     2.0425,   2.8615,   3.4412,   1.0000,   1.0000,   1.0000,
     2.3669,   2.3669,   7.1792,   7.1792,   7.2195,   7.2195,
     4.8045,   4.8045,   4.7461,   4.3065,  27.1492,  22.8977,
    31.9030,  19.4535,   1.0000,   1.0000,   1.0000,   1.0000,
     1.2644,   1.2630,   1.8731,   1.8645,   2.0373,   1.4146,
     1.4441,   1.5381,   0.6377,   0.9082
  };

  std::unique_ptr<CLCommandQueue> queue = CLDeviceContext::MakeCommandQueue();
  std::unique_ptr<CLBuffer> lab_1_buffer(CLDeviceContext::MakeBuffer(
      kSWBTestDataCount * 4 * sizeof(float)));
  lab_1_buffer->EnqueueCopyToDevice(queue.get(), lab_1);
  std::unique_ptr<CLBuffer> lab_2_buffer(CLDeviceContext::MakeBuffer(
      kSWBTestDataCount * 4 * sizeof(float)));
  lab_2_buffer->EnqueueCopyToDevice(queue.get(), lab_2);
  std::unique_ptr<CLBuffer> out_buffer(CLDeviceContext::MakeBuffer(
      kSWBTestDataCount * sizeof(float)));
  std::unique_ptr<CLKernel> kernel(
      CLDeviceContext::MakeKernel(CLProgram::Programs::kCiede2k));
  kernel->SetBufferArgument(0, lab_1_buffer.get());
  kernel->SetBufferArgument(1, lab_2_buffer.get());
  kernel->SetBufferArgument(2, out_buffer.get());
  kernel->Enqueue(queue.get(), kSWBTestDataCount);
  std::unique_ptr<float[]> results(new float[kSWBTestDataCount]);
  out_buffer->EnqueueCopyFromDevice(queue.get(), results.get());
  queue->Finish();
  // Dropping to single precision can result in some substantially different
  // results from the published expected results, particularly for values very
  // near the arctan discontinuity, but we should always be within the same
  // ballpark. The following tolerances were derived experimentally.
  for (int i = 0; i < kSWBTestDataCount; ++i) {
    if (i == 9) {
      EXPECT_NEAR(expected_results[i], results[i], 0.05);
    } else if (i == 13) {
      EXPECT_NEAR(expected_results[i], results[i], 0.06);
    } else {
      EXPECT_NEAR(expected_results[i], results[i], 0.0001);
    }
  }
}

}  // namespace vcsmc
