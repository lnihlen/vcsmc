// Given input colors in L*a*b* color space returns the CIEDE2000 delta E value,
// a nonlinear color distance based on the human vision response.

// This is by no means an offical implementation of this algorithm.

// This implementation is with thanks to:
//
// "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary
// Test Data, and Mathematical Observations,", G. Sharma, W. Wu, E. N. Dalal,
// Color Research and Application, vol. 30. No. 1, February 2005.

// Note that CIEDE2000 is discontinuous so may not be useful for gradient
// descent algorithms.

__kernel void ciede_distance(__global __read_only float4* lab_1_row,
                             __global __read_only float4* lab_2_row,
                             __global __write_only float* delta_e_out) {
  int col = get_global_id(0);
  float4 lab_1 = lab_1_row[col];
  float4 lab_2 = lab_2_row[col];

  // Calculate C_1, h_1, and C_2, h_2.
  float C_star_1 = length((float2)(lab_1.y, lab_1.z));
  float C_star_2 = length((float2)(lab_2.y, lab_2.z));
  float C_star_mean = (C_star_1 + C_star_2) / 2;
  float G = 0.5 * (1 - sqrt(C_star_mean / (C_star_mean + 25.0)));
  float a_1 = (1 + G) * lab_1.y;
  float a_2 = (1 + G) * lab_2.y;
  float C_1 = length((float2)(a_prime_1, lab_1.z));
  float C_2 = length((float2)(a_prime_2, lab_2.z));
  float h_1 = (a_1 == 0.0 && lab_1.z == 0.0) ? 0.0 : atan2(lab_1.z, a_1);
  float h_2 = (a_2 == 0.0 && lab_2.z == 0.0) ? 0.0 : atan2(lab_2.z, a_2);

  // atan2 returns radians between (-pi, pi], remap to [0, 360) degrees.
  h_1 = (h_1 < 0 ? h_1 + (2.0 * M_PI_F) : h_1) * (180.0 / M_PI_F);
  h_2 = (h_2 < 0 ? h_2 + (2.0 * M_PI_F) : h_2) * (180.0 / M_PI_F);

  // Calculate del_L, del_C, del_H.
  float del_L = lab_2.x - lab_1.x;
  float del_C = C_2 - C_1;
  float del_h = 0.0;
  float C_product = C_1 * C_2;
  float h_diff = h_2 - h_1;
  if (C_product != 0.0) {
    if (h_diff < -180.0) {
      del_h = h_diff + 360.0;
    } else if (h_diff <= 180.0) {
      del_h = h_diff;
    } else {
      del_h = h_diff - 360.0;
    }
  }
  float del_H = 2.0 * sqrt(C_product) * sin((del_h * (M_PI_F / 180.0)) / 2.0);


  // Calculate CIEDE2000 Color-Difference Delta E00:
  float L_mean = (lab_1.x + lab_2.x) / 2.0;
  float C_mean = (C_1 + C_2) / 2.0;
  float h_mean = h_1 + h_2;
  float h_diff_abs = abs(h_diff);
  if (C_product != 0.0) {
    if (h_diff_abs < 180.0) {
      h_mean = h_mean / 2.0;
    } else if (h_mean < 360.0) {
      h_mean = (h_mean + 360.0) / 2.0;
    } else {
      h_mean = (h_mean - 360.0) / 2.0;
    }
  }

  float T = 1.0 - (0.17 * cos((h_mean - 30.0) * (M_PI_F / 180.0))) +
                  (0.24 * cos(h_mean * (M_PI_F / 180.0))) +
                  (0.32 * cos(((3.0 * h_mean) + 6.0) * (M_PI_F / 180.0))) -
                  (0.20 * cos(((4.0 * h_mean) - 63.0) * (M_PI_F / 180.0)));

  float del_theta = 30.0 * exp(-1.0 * pow((h_mean - 275.0) / 25.0, 2.0);
  float R_c = 2.0 * sqrt(C_mean / (C_mean + 25.0));
  // intermediate term L_int not part of standard
  float L_int = pow(L_mean - 50.0, 2.0);
  float S_L = 1.0 + ((0.015 * L_int) / sqrt(20.0 + L_int));
  float S_C = 1.0 + 0.045 * C_mean;
  float S_H = 1.0 + 0.015 * C_mean * T;
  float R_T = -1.0 * sin(2.0 * del_theta * (M_PI_F / 180.0));
  delta_e_out[col] = sqrt(pow(del_L / S_L, 2.0) +
                          pow(del_C / S_C, 2.0) +
                          pow(del_H / S_H, 2.0) +
                          R_T * (del_C / S_C) * (del_H / S_H));
}
