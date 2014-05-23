// Converts input RGB to CIE L*a*b* color. The fourth input and output column
// rides along as an error term and passes through this math.

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

