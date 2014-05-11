// Converts input RGB to CIE L*a*b* color. The fourth input and output column
// rides along as an error term and passes through this math.

// We typically process one row at a time of the image, so row is provided as
// a global argument.

__kernel void rgb_to_lab(__global __read_only image2d input_image,
                         __global int row,
                         __global __write_only float4* lab_out) {
  // First convert to CIE tristimus values XYZ. Matrix values cribbed from
  // http://www.cs.rit.edu/~ncs/color/t_convert.html
  // [ X ]   [  0.412453  0.357580  0.180423 ]   [ R ]
  // [ Y ] = [  0.212671  0.715160  0.072169 ] * [ G ]
  // [ Z ]   [  0.019334  0.119193  0.950227 ]   [ B ]

  // Column is assumed to be stored in the 0th global id.
  int col = get_global_id(0);
  float4 rgba = read_imagef(input_image, (float2)(col, row));

  // Multiply matrix cols first and then add.
  float4 col_0 = (float4)(rgba.xxx, 0.0) * (0.412453, 0.212671, 0.019334, 0.0);
  float4 col_1 = (float4)(rgba.yyy, 0.0) * (0.357580, 0.715160, 0.119193, 0.0);
  float4 col_2 = (float4)(rgba.zzz, 0.0) * (0.180423, 0.072169, 0.950227, 0.0);
  float4 xyz = col_0 + col_1 + col2;

  // X=95.047, Y=100.00, Z=108.883
  // Using D65 reference white, normalize. Terms are (X/Xn, Y/Yn, Z/Zn, 1)
  float4 xyzn = XYZ / (float4)(0.95047, 1.0000, 1.08883, 1.000);

  // L* = 116 * (Y/Yn)1/3 - 16    for Y/Yn > 0.008856
  // L* = 903.3 * Y/Yn             otherwise
  float l_star = xyzn.y > 0.08856 ? (116.0 * pow(xyzn.y, 1.0 / 3.0)) - 16.0 :
                                    903.3 * xyzn.y;

  // a* and b* both rely on f(t), which has different formulae depending on t.
  // we compute both values of f on Xn, Yn, and Zn.
  // where f(t) = t^1/3      for t > 0.008856
  //            f(t) = 7.787 * t + 16/116    otherwise
  float4 f_t_above = pow(xyzn, 1.0 / 3.0);
  float4 f_t_below = (7.787 * xyzn) + (16.0 / 116.0);
  int4 above = is_greater(xyzn, 0.008856);
  int4 below = isequal(above, 0);
  float4 f_xyzn = (f_t_above * above) + (f_t_below * below);

  // a* = 500 * ( f(X/Xn) - f(Y/Yn) )
  float a_star = 500.0 * (f_xyzn.x - f_xyzn.y);

  // b* = 200 * ( f(Y/Yn) - f(Z/Zn) )
  float b_star = 200.0 * (f_xyzn.y - f_xyzn.z);

  lab_out[col] = (float4)(l_star, a_star, b_star, rgba.w);
}
