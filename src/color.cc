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

void RGBAToLab(const uint8* rgba, float* lab) {
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

// Conversion detailed in Wikipedia
// https://en.wikipedia.org/wiki/YUV, for HDTV BT.709
// adjusted slightly for better normalization.
void RGBAToNormalizedYuv(const uint8* rgba, float* nyuv) {
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
  nyuv[0] = y;
  nyuv[1] = (u + kUOffset) / kURange;
  nyuv[2] = (v + kVOffset) / kVRange;
  nyuv[3] = 1.0f;
}

float Ciede2k(const float* laba_1, const float* laba_2) {
  float L_star_1 = laba_1[0];
  float a_star_1 = laba_1[1];
  float b_star_1 = laba_1[2];

  float L_star_2 = laba_2[0];
  float a_star_2 = laba_2[1];
  float b_star_2 = laba_2[2];

  float C_star_1 = sqrt((a_star_1 * a_star_1) + (b_star_1 * b_star_1));
  float C_star_2 = sqrt((a_star_2 * a_star_2) + (b_star_2 * b_star_2));
  float C_star_mean_pow = pow((C_star_1 + C_star_2) / 2.0, 7.0);
  float G = 0.5 * (1.0 - sqrt(C_star_mean_pow /
                               (C_star_mean_pow + 6103515625.0)));  // 25^7
  float a_1 = (1.0 + G) * a_star_1;
  float a_2 = (1.0 + G) * a_star_2;
  float C_1 = sqrt((a_1 * a_1) + (b_star_1 * b_star_1));
  float C_2 = sqrt((a_2 * a_2) + (b_star_2 * b_star_2));
  float h_1 = (a_1 == 0.0 && b_star_1 == 0.0) ? 0.0 : atan2(b_star_1, a_1);
  float h_2 = (a_2 == 0.0 && b_star_2 == 0.0) ? 0.0 : atan2(b_star_2, a_2);
  h_1 = (h_1 < 0.0 ? h_1 + (2.0 * kPi) : h_1) * (180.0 / kPi);
  h_2 = (h_2 < 0.0 ? h_2 + (2.0 * kPi) : h_2) * (180.0 / kPi);

  float del_L = L_star_2 - L_star_1;
  float del_C = C_2 - C_1;
  float C_product = C_1 * C_2;
  float h_diff = h_2 - h_1;
  float del_h = 0.0;
  if (C_product != 0.0) {
    if (h_diff < -180.0) {
      del_h = h_diff + 360.0;
    } else if (h_diff <= 180.0) {
      del_h = h_diff;
    } else {
      del_h = h_diff - 360.0;
    }
  }

  float del_H = 2.0 * sqrt(C_product) * sin((del_h * (kPi / 180.0)) / 2.0);

  float L_mean = (L_star_1 + L_star_2) / 2.0;
  float C_mean = (C_1 + C_2) / 2.0f;
  float h_sum = h_1 + h_2;
  float h_mean = h_sum;
  float h_abs = std::abs(h_1 - h_2);
  if (C_product != 0.0) {
    if (h_abs <= 180.0) {
      h_mean = h_sum / 2.0;
    } else if (h_sum < 360.0) {
      h_mean = (h_sum + 360.0) / 2.0;
    } else {
      h_mean = (h_sum - 360.0) / 2.0;
    }
  }

  float T = 1.0 - (0.17 * cos((h_mean - 30.0) * (kPi / 180.0))) +
                   (0.24 * cos(2.0 * h_mean * (kPi / 180.0))) +
                   (0.32 * cos(((3.0 * h_mean) + 6.0) * (kPi / 180.0))) -
                   (0.20 * cos(((4.0 * h_mean) - 63.0) * (kPi / 180.0)));

  float del_theta = 30.0 * exp(-1.0 * pow((h_mean - 275.0) / 25.0, 2.0));
  float C_mean_pow = pow(C_mean, 7.0);
  float R_c = 2.0 * sqrt(C_mean_pow / (C_mean_pow +  6103515625.0));

  // Intermediate term L_int not part of standard formulation.
  float L_int = pow(L_mean - 50.0, 2.0);
  float S_L = 1.0 + ((0.015 * L_int) / sqrt(20.0 + L_int));
  float S_C = 1.0 + 0.045 * C_mean;
  float S_H = 1.0 + 0.015 * C_mean * T;
  float R_T = -1.0 * sin(2.0 * del_theta * (kPi / 180.0)) * R_c;
  float delta_e_out = sqrt(pow(del_L / S_L, 2.0) +
                            pow(del_C / S_C, 2.0) +
                            pow(del_H / S_H, 2.0) +
                            R_T * (del_C / S_C) * (del_H / S_H));
  return delta_e_out;
}

}  // namespace vcsmc
