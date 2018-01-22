#include "color.h"

#include <cmath>
#include <cstdlib>

#include "constants.h"

namespace {

inline float gamma_compand(float V) {
  return V <= 0.04045 ? V / 12.92 : powf((V + 0.055) / 1.055, 2.4);
}

const float kEpsilon = 216.0 / 24389.0;
const float kKappa = 24389.0 / 27.0;

inline float f(float x) {
  return x > kEpsilon ? powf(x, 1.0 / 3.0) : ((kKappa * x) + 16.0) / 116.0;
}

}

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

void RgbaToLaba(const uint8* rgba, float* lab) {
  float r = gamma_compand(static_cast<float>(rgba[0]) / 255.0);
  float g = gamma_compand(static_cast<float>(rgba[1]) / 255.0);
  float b = gamma_compand(static_cast<float>(rgba[2]) / 255.0);

  // D65 reference white, Xr=0.95047, Yr=1.0, Zr=1.08883
  float X = (0.4124564 * r) + (0.3575761 * g) + (0.1804375 * b);
  float Y = (0.2126729 * r) + (0.7151522 * g) + (0.0721750 * b);
  float Z = (0.0193339 * r) + (0.1191920 * g) + (0.9503041 * b);
  float x = X / 0.95047;
  float y = Y;
  float z = Z / 1.08883;

  float f_x = f(x);
  float f_y = f(y);
  float f_z = f(z);

  float L_star = (116.0 * f_y) - 16.0;
  float a_star = 500.0 * (f_x - f_y);
  float b_star = 200.0 * (f_y - f_z);

  lab[0] = L_star;
  lab[1] = a_star;
  lab[2] = b_star;
}

}  // namespace vcsmc
