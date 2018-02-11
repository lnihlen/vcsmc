#include "ciede_2k.h"

#include "constants.h"

namespace vcsmc {

__global__ void Ciede2k(const float4* laba_a_in,
                        const float4* laba_b_in,
                        float* ciede_2k_out) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = (y * kTargetFrameWidthPixels) + x;

  float4 laba_a = laba_a_in[index];
  float4 laba_b = laba_b_in[index];

  float L_star_1 = laba_a.x;
  float a_star_1 = laba_a.y;
  float b_star_1 = laba_a.z;

  float L_star_2 = laba_b.x;
  float a_star_2 = laba_b.y;
  float b_star_2 = laba_b.z;

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

  ciede_2k_out[index] = delta_e_out / kMaxCiede2kDistance;
}

}
