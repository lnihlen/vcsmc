#include "cl_program.h"

#include <assert.h>

#define CL_PROGRAM(src) #src

namespace vcsmc {

// static
std::string CLProgram::GetProgramString(Programs program) {
  switch (program) {
    case kCiede2k:
      return CL_PROGRAM(
// Given input colors in L*a*b* color space returns the CIEDE2000 delta E value,
// a nonlinear color distance based on the human vision response.

// This is by no means an offical implementation of this algorithm.

// This implementation is with thanks to:
//
// "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary
// Test Data, and Mathematical Observations,", G. Sharma, W. Wu, E. N. Dalal,
// Color Research and Application, vol. 30. No. 1, February 2005.

// Note that CIEDE2000 is discontinuous so is not useful for gradient
// descent algorithms.

__kernel void ciede2k(__global __read_only float4* lab_1_row,
                      __global __read_only float4* lab_2_row,
                      __global __write_only float* delta_e_out) {
  int col = get_global_id(0);
  float4 lab_1 = lab_1_row[col];
  float4 lab_2 = lab_2_row[col];

  // Calculate C_1, h_1, and C_2, h_2.
  float C_star_1 = length(lab_1.yz);
  float C_star_2 = length(lab_2.yz);
  float C_star_mean_pow = pow((C_star_1 + C_star_2) / 2.0f, 7.0f);
  float G = 0.5f * (1.0f - sqrt(C_star_mean_pow /
                               (C_star_mean_pow + 6103515625.0f))); // 25^7
  float a_1 = (1.0f + G) * lab_1.y;
  float a_2 = (1.0f + G) * lab_2.y;
  float C_1 = length((float2)(a_1, lab_1.z));
  float C_2 = length((float2)(a_2, lab_2.z));
  float h_1 = (a_1 == 0.0f && lab_1.z == 0.0f) ? 0.0f : atan2(lab_1.z, a_1);
  float h_2 = (a_2 == 0.0f && lab_2.z == 0.0f) ? 0.0f : atan2(lab_2.z, a_2);

  // atan2 returns radians between (-pi, pi], remap to [0, 360) degrees.
  h_1 = (h_1 < 0.0f ? h_1 + (2.0f * M_PI_F) : h_1) * (180.0f / M_PI_F);
  h_2 = (h_2 < 0.0f ? h_2 + (2.0f * M_PI_F) : h_2) * (180.0f / M_PI_F);

  // Calculate del_L, del_C, del_H.
  float del_L = lab_2.x - lab_1.x;
  float del_C = C_2 - C_1;
  float C_product = C_1 * C_2;
  float h_diff = h_2 - h_1;

  float del_h = 0.0f;
  if (C_product != 0.0f) {
    if (h_diff < -180.0f) {
      del_h = h_diff + 360.0f;
    } else if (h_diff <= 180.0f) {
      del_h = h_diff;
    } else {
      del_h = h_diff - 360.0f;
    }
  }

  float del_H = 2.0f * sqrt(C_product) *
                sin((del_h * (M_PI_F / 180.0f)) / 2.0f);

  // Calculate CIEDE2000 Color-Difference Delta E00:
  float L_mean = (lab_1.x + lab_2.x) / 2.0f;
  float C_mean = (C_1 + C_2) / 2.0f;
  float h_sum = h_1 + h_2;
  float h_mean = h_sum;
  float h_abs = fabs(h_1 - h_2);  // note reverse order from h_diff
  if (C_product != 0.0f) {
    if (h_abs <= 180.0f) {
      h_mean = h_sum / 2.0f;
    } else if (h_sum < 360.0f) {
      h_mean = (h_sum + 360.0f) / 2.0f;
    } else {
      h_mean = (h_sum - 360.0f) / 2.0f;
    }
  }

  float T = 1.0f - (0.17f * cos((h_mean - 30.0f) * (M_PI_F / 180.0f))) +
                   (0.24f * cos(2.0f * h_mean * (M_PI_F / 180.0f))) +
                   (0.32f * cos(((3.0f * h_mean) + 6.0f) * (M_PI_F / 180.0f))) -
                   (0.20f * cos(((4.0f * h_mean) - 63.0f) * (M_PI_F / 180.0f)));

  float del_theta = 30.0f * exp(-1.0f * pow((h_mean - 275.0f) / 25.0f, 2.0f));
  float C_mean_pow = pow(C_mean, 7.0f);
  float R_c = 2.0f * sqrt(C_mean_pow / (C_mean_pow +  6103515625.0f));

  // intermediate term L_int not part of standard
  float L_int = pow(L_mean - 50.0f, 2.0f);
  float S_L = 1.0f + ((0.015f * L_int) / sqrt(20.0f + L_int));
  float S_C = 1.0f + 0.045f * C_mean;
  float S_H = 1.0f + 0.015f * C_mean * T;
  float R_T = -1.0f * sin(2.0f * del_theta * (M_PI_F / 180.0f)) * R_c;
  delta_e_out[col] = sqrt(pow(del_L / S_L, 2.0f) +
                          pow(del_C / S_C, 2.0f) +
                          pow(del_H / S_H, 2.0f) +
                          R_T * (del_C / S_C) * (del_H / S_H));
}
      );

    case kRGBToLab:
      return CL_PROGRAM(
// Converts input RGB to CIE L*a*b* color. The fourth input and output column
// ride along as an error term and pass through this math.

// We typically process one row at a time of the image, so row is provided as
// a global argument.

// Notes, values, and algorithm from:
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

__kernel void rgb_to_lab(__read_only image2d_t input_image,
                         __read_only int image_row,
                         __global __write_only float4* lab_out) {
  int col = get_global_id(0);
  float4 rgba = read_imagef(input_image, (int2)(col, image_row));

  float4 v_below = rgba / 12.92f;
  float4 v_above = pow((rgba + 0.055f) / 1.055f, 2.4f);
  int4 below = islessequal(rgba, 0.04045f);
  float4 rgba_norm = select(v_above, v_below, below);

  // |X|   |0.4124564  0.3575761  0.1804375|   |R|
  // |Y| = |0.2126729  0.7151522  0.0721750| * |G|
  // |Z|   |0.0193339  0.1191920  0.9503041|   |B|
  float4 col_0 = (float4)(rgba_norm.xxx, 0.0f) *
      (float4)(0.4124564f, 0.2126729f, 0.0193339f, 0.0f);
  float4 col_1 = (float4)(rgba_norm.yyy, 0.0f) *
      (float4)(0.3575761f, 0.7151522f, 0.1191920f, 0.0f);
  float4 col_2 = (float4)(rgba_norm.zzz, 0.0f) *
      (float4)(0.1804375f, 0.0721750f, 0.9503041f, 0.0f);
  float4 xyz = col_0 + col_1 + col_2;

  // Xr=0.95047, Yr=1.0, Zr=1.08883, D65 reference white
  float4 xyzn = xyz / (float4)(0.95047f, 1.0f, 1.08883f, 1.0f);

  float4 f_below = (7.787f * xyzn) + (16.0f / 116.0f);
  float4 f_above = pow(xyzn, 1.0f / 3.0f);
  below = islessequal(xyzn, 216.0f / 24389.0f);
  float4 f = select(f_above, f_below, below);
  float4 Lab = (float4)((116.0f * f.y) - 16.0f,
                        500.0f * (f.x - f.y),
                        200.0f * (f.y - f.z),
                        rgba.w);
  lab_out[col] = Lab;
}
      );

    default:
      assert(false);
      return "";
  }
}

// static
std::string CLProgram::GetProgramName(Programs program) {
  switch (program) {
    case kCiede2k:
      return "ciede2k";

    case kRGBToLab:
      return "rgb_to_lab";

    default:
      assert(false);
      return "";
  }
}

}  // namespace vcsmc
