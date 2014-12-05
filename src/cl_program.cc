#include "cl_program.h"

#include <assert.h>

#define CL_PROGRAM(src) #src

namespace vcsmc {

// static
std::string CLProgram::GetProgramString(Programs program) {
  switch (program) {

//
// kCiede2k ===================================================================
//

    case kCiede2k:
      return CL_PROGRAM(
// Given input colors in L*a*b* color space returns the CIEDE2000 delta E value,
// a nonlinear color distance based on the human vision response.

// This is by no means an official implementation of this algorithm.

// This implementation is with thanks to:
//
// "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary
// Test Data, and Mathematical Observations,", G. Sharma, W. Wu, E. N. Dalal,
// Color Research and Application, vol. 30. No. 1, February 2005.

// Note that CIEDE2000 is discontinuous so is not useful for gradient
// descent algorithms.

// For inputs |pixel_lab_in| is pointing at some large array of pixel values
// and it is assumed that there is an offset applied to the global id.  The
// |reference_lab_in| points at the reference colors. This function will store
// the output at the row given by the index into the reference color table and
// the column by the pixel row with offset removed. It is assumed that the
// distances between every pixel color within the work dimensions described by
// |pixel_lab_in| and every color in |reference_lab_in| are to be computed.
__kernel void ciede2k(__global __read_only float4* pixel_lab_in,
                      __global __read_only float4* reference_lab_in,
                      __global __write_only float* delta_e_out) {
  int pixel_id = get_global_id(0);
  int reference_id = get_global_id(1);
  int output_id = (reference_id * get_global_size(1)) +
      (pixel_id - get_global_offset(0));
  float4 lab_1 = pixel_lab_in[pixel_id];
  float4 lab_2 = reference_lab_in[reference_id];

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
  delta_e_out[output_id] = sqrt(pow(del_L / S_L, 2.0f) +
                                pow(del_C / S_C, 2.0f) +
                                pow(del_H / S_H, 2.0f) +
                                R_T * (del_C / S_C) * (del_H / S_H));
}
      );  // end of kCiede2K

//
// kConvolve ==================================================================
//
    case kConvolve:
      return CL_PROGRAM(
// Convolves a square input |kernel| of dimensions |kernel_size| against the
// |input|, which is assumed to be of dimensions provided by global size, and
// saving the result in |output|.
__kernel void convolve(__global __read_only float* input,
                       __global __read_only float* c_kernel,
                       __read_only int kernel_size,
                       __global __write_only float* output) {
  int input_width = get_global_size(0);
  int input_height = get_global_size(1);
  int input_x = get_global_id(0);
  int input_y = get_global_id(1);

  // Start convolution on upper-right hand corner of kernel and continue
  // until reaching lower-right hand corner.
  int center_offset = (input_y * input_width) + input_x;
  int input_offset = center_offset -
      (kernel_size / 2) - ((kernel_size / 2) * input_width);
  int input_size = input_width * input_height;
  int kernel_offset = 0;
  float accum = 0.0f;
  for (int i = 0; i < kernel_size; ++i) {
    for (int j = 0; j < kernel_size; ++j) {
      if (input_offset >= 0 && input_offset < input_size) {
        accum += input[input_offset] * c_kernel[kernel_offset];
      }
      ++kernel_offset;
      ++input_offset;
    }
    // skip to next row of input kernel.
    input_offset += input_width - (1 + kernel_size);
  }

  output[center_offset] = accum;
}
      );  // end of kConvolve

//
// kDownsampleErrors ==========================================================
//

    case kDownsampleErrors:
      return CL_PROGRAM(
// Given an input float array of error distances |input_errors| of |input_width|
// fills an output array with interpolated distances. Width of output is assumed
// to be equal to the x dimension of the global work group.
__kernel void downsample_errors(__global __read_only float* input_errors,
                                __read_only int input_width,
                                __global __write_only float* output_errors) {
  int output_col = get_global_id(0);
  int output_row = get_global_id(1);
  int output_width = get_global_size(0);
  int input_row_offset = output_row * input_width;
  // It is possible the work group is of size related to the |input_width| not
  // the |output_width|, if so we exit.
  if (output_col >= input_width)
    return;
  float start_pixel = ((float)output_col * (float)input_width) /
      (float)output_width;
  float end_pixel = (((float)output_col + 1.0f) * (float)input_width) /
      (float)output_width;
  float first_whole_pixel = ceil(start_pixel);
  int first_whole_pixel_int = (int)first_whole_pixel + input_row_offset;
  float last_fractional_pixel = floor(end_pixel);
  int last_fractional_pixel_int = (int)last_fractional_pixel + input_row_offset;
  float total_error = 0.0f;

  // Calculate whole pixel error
  for (int i = first_whole_pixel_int; i < last_fractional_pixel_int; ++i)
    total_error += input_errors[i];

  // Include left-side fractional error
  total_error += input_errors[(int)floor(start_pixel) + input_row_offset] *
      (first_whole_pixel - start_pixel);

  // Right-side fractional error. It's possible that due to roundoff error the
  // last_fractional_pixel may equal output_width, if so we ignore.
  if (last_fractional_pixel_int - input_row_offset < input_width) {
    total_error += input_errors[last_fractional_pixel_int] *
        (end_pixel - last_fractional_pixel);
  }

  output_errors[(output_row * output_width) + output_col] = total_error;
}
      );  // end of kDownsampleErrors

//
// kFFTRadix2 =================================================================
//

    case kFFTRadix2:
      return CL_PROGRAM(
// Computes a Fast Fourier Transform of the supplied complex data. |input_data|
// is packed as float2s (real, complex). Since the radix-2 FFT combines k
// subsequences of length m to k/2 subsequences of 2m each iteration
// |subsequence_size| should count like 1, 2, 4, 8, .. N/2 on successive calls
// to this kernel, for N points. N must be a power of two. This kernel must be
// run with N/2 threads.

// Code largely cribbed from Eric Bainville's excellent article on OpenCL FFT
// at http://www.bealto.com/gpu-fft2_opencl-1.html.

__kernel void fft_radix_2(__global __read_only float2* input_data,
                          __read_only int subsequence_size,
                          __read_only int output_stride,
                          __global __write_only float2* output_data) {
  // |i| is our thread index, should range in [0, N/2).
  int i = get_global_id(0);
  // |y| is vertical thread index, range in [0, image_height)
  int y = get_global_id(1);
  int t = get_global_size(0); // must be equal to N/2
  int input_offset = y * t * 2;

  input_data += input_offset;
  float2 i0 = input_data[i];
  float2 i1 = input_data[i + t];

  // |k| is the index within the input subsequence, in [0, subsequence_size).
  int k = i & (subsequence_size - 1);

  // Twiddle second input.
  float cs;
  float sn = sincos(-3.1415926979f * (float)k / (float)subsequence_size, &cs);
  i1 = (float2)((i1.x * cs) - (i1.y * sn), (i1.x * sn) + (i1.y * cs));

  // Perform radix-2 DFT.
  float2 temp = i0 - i1;
  i0 += i1;
  i1 = temp;

  // Write output, |j| is our output index.
  int j = ((i - k) << 1) + k;
  int output_offset = output_stride > 1 ? y : input_offset;
  output_data += output_offset;
  output_data[j * output_stride] = i0;
  output_data[(j + subsequence_size) * output_stride] = i1;
}
      );  // end of kFFTRadix2

//
// kHistogramClasses ==========================================================
//

    case kHistogramClasses:
      return CL_PROGRAM(
// |scratch| needs to be num_classes * width in size.
__kernel void histogram_classes(__global __read_only uint* classes,
                                __read_only uint num_classes,
                                __local uint* scratch,
                                __global __write_only uint* counts) {
  uint pixel = get_global_id(0);
  uint width = get_global_size(0);
  uint npot_width = (uint)pown(2.0f, ilogb((float)width) + 1);
  uint scratch_index = pixel * num_classes;
  for (uint i = 0; i < num_classes; ++i)
    scratch[scratch_index + i] = 0;
  scratch[scratch_index + classes[pixel]] = 1;

  // Reduce counts to final output.
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = npot_width / 2; offset > 0; offset = offset / 2) {
    uint their_index = scratch_index + (offset * num_classes);
    if (pixel < offset && pixel + offset < width) {
      for (uint i = 0; i < num_classes; ++i) {
        uint their_count = scratch[their_index + i];
        uint my_count = scratch[scratch_index + i];
        scratch[scratch_index + i] = their_count + my_count;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (pixel < num_classes)
    counts[pixel] = scratch[pixel];
}
      );  // end of kHistogramClasses

//
// kInverseFFTNormalize =======================================================
//

    case kInverseFFTNormalize:
      return CL_PROGRAM(
// An inverse FFT can be performed by conjugating each input complex number,
// then doing a forward FFT on the conjugated data, then conjugating again and
// scaling the data by 1/N. This shader can be applied before and after a
// series of forward FFTs to perform the before and after conjugation.
__kernel void inverse_fft_normalize(__global __read_only float4* input_data,
                                    __read_only float norm,
                                    __global __write_only float4* output_data) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int t = get_global_size(0);  // must be equal to width in units of float4s
  int offset = j * t;
  float4 ic = input_data[i + offset];
  float4 oc = (float4)(ic.x, -ic.y, ic.z, -ic.w);
  output_data[i + offset] = oc / norm;
}
      );  // end of kInverseFFTNormalize

//
// kKMeansClassify =============================================================
//

    case kKMeansClassify:
      return CL_PROGRAM(
// Given a 2D array of color distances with width global_size(0) and height of
// kNTSCColors at |color_errors|, and an array of currently selected |colors| of
// length |num_classes|, this shader will determine the lowest error color at
// each pixel and store its index into |classes|.
__kernel void k_means_classify(__global __read_only float* color_errors,
                               __global __read_only uint* colors,
                               __read_only uint num_classes,
                               __global __write_only uint* classes) {
  uint pixel = get_global_id(0);
  uint width = get_global_size(0);
  float min_error = color_errors[(colors[0] * width) + pixel];
  uint min_error_index = 0;
  for (uint i = 1; i < num_classes; ++i) {
    float error = color_errors[(colors[i] * width) + pixel];
    if (error < min_error) {
      min_error = error;
      min_error_index = i;
    }
  }
  classes[pixel] = min_error_index;
}
      );  // end of kKMeansClassify

//
// kKMeansColor ===============================================================
//

    case kKMeansColor:
      return CL_PROGRAM(
// Run one thread for each color. |error_scratch| needs to be number of possible
// colors times |num_classes| in size, and |class_scratch| needs to be
// the same. Note that the number of possible colors must be a power of 2.
__kernel void k_means_color(__global __read_only float* color_errors,
                            __global __read_only uint* classes,
                            __read_only uint image_width,
                            __read_only uint num_classes,
                            __read_only uint iteration,
                            __local float* error_scratch,
                            __local uint* class_scratch,
                            __global __write_only float* fit_error,
                            __global __write_only uint* colors) {
  uint color_id = get_global_id(0);
  uint num_colors = get_global_size(0);
  // The |scratch| arrays are {num_colors| rows of |num_classes| elements.
  int scratch_index = (num_classes * color_id);

  // Zero out scratch arrays for each class error counter for this color.
  for (uint i = 0; i < num_classes; ++i) {
    error_scratch[scratch_index + i] = 0.0f;
    class_scratch[scratch_index + i] = color_id;
  }

  // Advance error table pointer to our color.
  color_errors += image_width * color_id;
  // Sum up error for our color for each class of pixel.
  for (uint i = 0; i < image_width; ++i)
    error_scratch[scratch_index + classes[i]] += color_errors[i];

  // Now we reduce to find min error color for each class.
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = num_colors / 2; offset > 0; offset = offset / 2) {
    uint their_index = scratch_index + (offset * num_classes);
    if (color_id < offset) {
      for (uint i = 0; i < num_classes; ++i) {
        float my_error = error_scratch[scratch_index + i];
        float their_error = error_scratch[their_index + i];
        if (their_error < my_error) {
          error_scratch[scratch_index + i] = their_error;
          class_scratch[scratch_index + i] = class_scratch[their_index + i];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Use the lowest |num_classes| threads to copy final class reductions to
  // the output.
  if (color_id < num_classes)
    colors[color_id] = class_scratch[color_id];

  if (color_id == 0) {
    float total_error = 0.0f;
    for (uint i = 0; i < num_classes; ++i) {
      total_error += error_scratch[i];
    }
    fit_error[iteration] = total_error;
  }
}
      );  // end of kKMeansColor
//
// kMakeBitmap ================================================================
//

    case kMakeBitmap:
      return CL_PROGRAM(
// TODO: write comment
__kernel void make_bitmap(__global __read_only float* input,
                          __global __read_only float* mean,
                          __global __read_only float* std_dev,
                          __global __write_only uchar* output) {
  int i = get_global_id(0);
  float inf = input[i];
  float threshold = *mean + *std_dev;
  output[i] = inf >= threshold ? 0xff : 0x00;
}
      );  // end of kMakeBitmap

//
// kMean ======================================================================
//

    case kMean:
      return CL_PROGRAM(
// Parallel reduction code. |in| points to |length| floats which are to be
// summed. |out| points to a single float for mean. Run this with global
// work group size equal to the max allowed in a local work group. |local|
// points to scratch storage sufficient to hold one float for each thread.
__kernel void mean(__global __read_only float* in,
                   __read_only int length,
                   __local float* scratch,
                   __global __write_only float* out) {
  int number_of_threads = get_global_size(0);
  int thread_index = get_global_id(0);
  int input_index = thread_index;
  float accum = 0.0f;

  // Serial addition of input array to reduce to |number_of_threads| sums.
  while (input_index < length) {
    accum += in[input_index];
    input_index += number_of_threads;
  }

  scratch[thread_index] = accum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = number_of_threads / 2; offset > 0; offset = offset / 2) {
    if (thread_index < offset) {
      float other = scratch[thread_index + offset];
      float mine = scratch[thread_index];
      scratch[thread_index] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (thread_index == 0) {
    *out = scratch[0] / (float)length;
  }
}
      );  // end of kMean

//
// kPackComplexToReal =========================================================
//

    case kPackComplexToReal:
      return CL_PROGRAM(
__kernel void pack_complex_to_real(__global __read_only float2* in_complex,
                                   __read_only int input_width,
                                   __read_only int input_height,
                                   __read_only int output_width,
                                   __read_only int output_height,
                                   __global __write_only float* out_real) {
  int number_of_threads = get_global_size(0);
  int output_index = get_global_id(0);
  int output_length = output_width * output_height;

  while (output_index < output_length) {
    int input_row = output_index / output_width;
    int input_col = output_index % output_width;
    int input_index = (input_row * input_width) + input_col;
    out_real[output_index] = length(in_complex[input_index]);
    output_index += number_of_threads;
  }
}
      );  // end of kPackComplexToReal


//
// kRGBToLab ==================================================================
//

    case kRGBToLab:
      return CL_PROGRAM(
// Converts input RGB to CIE L*a*b* color.
// Notes, values, and algorithm from:
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
__kernel void rgb_to_lab(__read_only image2d_t input_image,
                         __global __write_only float4* lab_out) {
  int col = get_global_id(0);
  int row = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);

  // OpenCL reads images in the opposite orientation of what we expect, so we
  // flip the image vertically.
  float4 rgba = read_imagef(input_image, (int2)(col, height - row - 1));
  float3 rgb_in = rgba.xyz;

  float3 v_below = rgb_in / 12.92f;
  float3 v_above = pow((rgb_in + 0.055f) / 1.055f, 2.4f);
  int3 below = islessequal(rgb_in, 0.04045f);
  float3 rgb_norm = select(v_above, v_below, below);

  // |X|   |0.4124564  0.3575761  0.1804375|   |R|
  // |Y| = |0.2126729  0.7151522  0.0721750| * |G|
  // |Z|   |0.0193339  0.1191920  0.9503041|   |B|
  float3 col_0 = rgb_norm.xxx * (float3)(0.4124564f, 0.2126729f, 0.0193339f);
  float3 col_1 = rgb_norm.yyy * (float3)(0.3575761f, 0.7151522f, 0.1191920f);
  float3 col_2 = rgb_norm.zzz * (float3)(0.1804375f, 0.0721750f, 0.9503041f);
  float3 xyz = col_0 + col_1 + col_2;

  // Xr=0.95047, Yr=1.0, Zr=1.08883, D65 reference white
  float3 xyzn = xyz / (float3)(0.95047f, 1.0f, 1.08883f);

  float3 f_below = (7.787f * xyzn) + (16.0f / 116.0f);
  float3 f_above = pow(xyzn, 1.0f / 3.0f);
  below = islessequal(xyzn, 216.0f / 24389.0f);
  float3 f = select(f_above, f_below, below);
  float3 Lab = (float3)((116.0f * f.y) - 16.0f,
                        500.0f * (f.x - f.y),
                        200.0f * (f.y - f.z));

  lab_out[(row * width) + col] = (float4)(Lab.xyz, 1.0f);
}
      );  // end of kRGBToLab

//
// kSpectralResidual ==========================================================
//

    case kSpectralResidual:
      return CL_PROGRAM(
__kernel void spectral_residual(__global __read_only float2* input_data,
                                __global __write_only float2* output_data) {
  int col = get_global_id(0);
  int row = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);
  // center input data pointer around (col, row)
  int c = (row * width) + col;
  float2 center = input_data[c];
  float log_center = log(length(center));
  float2 log_avg = (float2)(log_center, 1.0f);

  // Take average of logarithms of 9 pixels centered at pixel 4:
  // 0 1 2
  // 3 4 5
  // 6 7 8
  //
  // 012 - upper row
  if (row > 0) {
    // 0 - upper right-hand corner
    if (col > 0) {
      log_avg += (float2)(log(length(input_data[c - width - 1])), 1.0f);
    }
    // 1 - upper center
    log_avg += (float2)(log(length(input_data[c - width])), 1.0f);
    // 2 - upper right-hand corner
    if (col < width - 1) {
      log_avg += (float2)(log(length(input_data[c - width + 1])), 1.0f);
    }
  }

  // 3 - left center
  if (col > 0) {
    log_avg += (float2)(log(length(input_data[c - 1])), 1.0f);
  }
  // 5 - right center
  if (col < width - 1) {
    log_avg += (float2)(log(length(input_data[c + 1])), 1.0f);
  }

  // 678 - bottom row
  if (row < height - 1) {
    // 6 - lower left-hand corner
    if (col > 0) {
      log_avg += (float2)(log(length(input_data[c + width - 1])), 1.0f);
    }
    // 7 - bottom center
    log_avg += (float2)(log(length(input_data[c + width])), 1.0f);
    // 8 - bottom right-hand corner
    if (col < width - 1) {
      log_avg += (float2)(log(length(input_data[c + width + 1])), 1.0f);
    }
  }

  float avg = log_avg.x / log_avg.y;
  float residual = exp(log_center - avg);
  float phi = atan2(center.y, center.x);
  float2 out = (float2)(residual * cos(phi), residual * sin(phi));
  output_data[c] = out;
}
      );  // end of kSpectralResidual

//
// kSquare ====================================================================
//

    case kSquare:
      return CL_PROGRAM(
__kernel void square(__global __read_only float* in,
                     __global __write_only float* out) {
  int i = get_global_id(0);
  float inf = in[i];
  out[i] = inf * inf;
}
      );  // end of kSquare

//
// kStandardDeviation =========================================================
//

    case kStandardDeviation:
      return CL_PROGRAM(
__kernel void standard_deviation(__global __read_only float* in,
                                 __global __read_only float* mean_in,
                                 __read_only int length,
                                 __local float* scratch,
                                 __global __write_only float* out) {
  int number_of_threads = get_global_size(0);
  int thread_index = get_global_id(0);
  int input_index = thread_index;
  float accum = 0.0f;
  float mean = *mean_in;

  // Serial addition of input array to reduce to |number_of_threads| sums.
  while (input_index < length) {
    accum += pow(in[input_index] - mean, 2.0f);
    input_index += number_of_threads;
  }

  scratch[thread_index] = accum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int offset = number_of_threads / 2; offset > 0; offset = offset / 2) {
    if (thread_index < offset) {
      float other = scratch[thread_index + offset];
      float mine = scratch[thread_index];
      scratch[thread_index] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (thread_index == 0) {
    *out = sqrt(scratch[0] / (float)length);
  }
}

      );   // end of kStandardDeviation

//
// kUnpackRealToComplex =======================================================
//

    case kUnpackRealToComplex:
      return CL_PROGRAM(
// Copies a input image of single floats to an output buffer of float2s of a
// larger size. This is used to expand real-valued packed image floats into a
// zero-padded power-of-two size for FFT, with real values and zero imaginary
// values interleaved. Should be run with global size of the maximum number of
// threads the OpenCL API says it can do in a single work group.
__kernel void unpack_real_to_complex(__global __read_only float* in_real,
                                     __read_only int input_width,
                                     __read_only int input_height,
                                     __read_only int output_width,
                                     __read_only int output_height,
                                     __global __write_only float2* out_cplx) {
  int number_of_threads = get_global_size(0);
  int output_index = get_global_id(0);
  int output_length = output_width * output_height;

  while (output_index < output_length) {
    int output_row = output_index / output_width;
    int output_col = output_index % output_width;
    if (output_row < input_height) {
      if (output_col < input_width) {
        int input_index = (output_row * input_width) + output_col;
        out_cplx[output_index] = (float2)(in_real[input_index], 0.0f);
      } else {
        out_cplx[output_index] = (float2)(0.0f, 0.0f);
      }
    } else {
      out_cplx[output_index] = (float2)(0.0f, 0.0f);
    }
    output_index += number_of_threads;
  }
}
      );  // end of kUnpackRealToComplex

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

    case kConvolve:
      return "convolve";

    case kDownsampleErrors:
      return "downsample_errors";

    case kFFTRadix2:
      return "fft_radix_2";

    case kHistogramClasses:
      return "histogram_classes";

    case kInverseFFTNormalize:
      return "inverse_fft_normalize";

    case kKMeansClassify:
      return "k_means_classify";

    case kKMeansColor:
      return "k_means_color";

    case kMakeBitmap:
      return "make_bitmap";

    case kMean:
      return "mean";

    case kPackComplexToReal:
      return "pack_complex_to_real";

    case kRGBToLab:
      return "rgb_to_lab";

    case kSpectralResidual:
      return "spectral_residual";

    case kSquare:
      return "square";

    case kStandardDeviation:
      return "standard_deviation";

    case kUnpackRealToComplex:
      return "unpack_real_to_complex";

    default:
      assert(false);
      return "";
  }
}

}  // namespace vcsmc
