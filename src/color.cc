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
