#include "color.h"

#include <cmath>
#include <cstdlib>

#include "constants.h"

namespace vcsmc {

// Conversion detailed in Wikipedia
// https://en.wikipedia.org/wiki/YUV, for HDTV BT.709
// adjusted slightly for better normalization.
void RgbaToNormalizedYuv(const uint8* rgba, float* nyuv) {
  float r = static_cast<float>(rgba[0]) / 255.0f;
  float g = static_cast<float>(rgba[1]) / 255.0f;
  float b = static_cast<float>(rgba[2]) / 255.0f;

  float y =  (0.2126  * r) + (0.7152  * g) + (0.0722  * b);
  float u = -(0.09991 * r) - (0.33609 * g) + (0.436   * b);
  float v =  (0.615   * r) - (0.55861 * g) + (0.05639 * b);

  // Without normalization:
  //
  //       |     R    |     G    |     B    |     Y    |     U     |     V     |
  // ------+----------+----------+----------+----------+-----------+-----------+
  // max Y | 1.000000 | 1.000000 | 1.000000 | 1.000000 |  0.000000 |  0.112780 |
  // min Y | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  0.000000 |  0.000000 |
  // max U | 0.000000 | 0.000000 | 1.000000 | 0.072200 |  0.436000 |  0.056390 |
  // min U | 1.000000 | 1.000000 | 0.000000 | 0.927800 | -0.436000 |  0.056390 |
  // max V | 1.000000 | 0.000000 | 1.000000 | 0.284800 |  0.336090 |  0.671390 |
  // min V | 0.000000 | 1.000000 | 0.000000 | 0.715200 | -0.336090 | -0.558610 |
  //
  // We are evaluating each element in isolation so don't need to worry about
  // disrupting the balance between elements, so can scale each value
  // differently to reach full range of [0, 1] to allow for MSSIM to work
  // correctly and accurately.
  const float kUOffset = 0.436f;
  const float kURange = 0.872f;
  const float kVOffset = 0.55861f;
  const float kVRange = 1.23f;

  // With normalization:
  //
  //       |     R    |     G    |     B    |     Y    |     U    |     V    |
  // ------+----------+----------+----------+----------+----------+----------+
  // max Y | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 0.545846 |
  // min Y | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.454154 |
  // max U | 0.000000 | 0.000000 | 1.000000 | 0.072200 | 1.000000 | 0.500000 |
  // min U | 1.000000 | 1.000000 | 0.000000 | 0.927800 | 0.000000 | 0.500000 |
  // max V | 1.000000 | 0.000000 | 1.000000 | 0.284800 | 0.885424 | 1.000000 |
  // min V | 0.000000 | 1.000000 | 0.000000 | 0.715200 | 0.114576 | 0.000000 |
  //
  nyuv[0] = y;
  nyuv[1] = (u + kUOffset) / kURange;
  nyuv[2] = (v + kVOffset) / kVRange;
  nyuv[3] = 1.0f;
}

}  // namespace vcsmc
