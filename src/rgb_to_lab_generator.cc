#include "Halide.h"

class RgbToLab : public Halide::Generator<RgbToLab> {
 public:
  Input<Halide::Buffer<uint8_t>> rgb_input{"rgb_input", 3};

  Output<Halide::Buffer<float>> lab_output{"lab_output", 3};

  void generate() {
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    // Cast input image per-channel to float.
    Halide::Func cast_to_float("cast_to_float");
    cast_to_float(x, y, c) = Halide::cast<float>(rgb_input(x, y, c)) / 255.0f;

    // Inverse sRGB Companding
    Halide::Func gamma_compand("gamma_compand");
    gamma_compand(x, y, c) = Halide::select(cast_to_float(x, y, c) <= 0.04045f,
        cast_to_float(x, y, c) / 12.92f,
        pow((cast_to_float(x, y, c) + 0.055f) / 1.055f, 2.4f));

    // Adjust to reference white, convert to XYZ color space.
    Halide::Func d65_white("d65_white");
    d65_white(x, y) = Halide::Tuple(
        ((gamma_compand(x, y, 0) * 0.4124564f) +
            (gamma_compand(x, y, 1) * 0.3575761f) +
            (gamma_compand(x, y, 2) * 0.1804375f)) / 0.95047f,
        (gamma_compand(x, y, 0) * 0.2126729f) +
            (gamma_compand(x, y, 1) * 0.7151522f) +
            (gamma_compand(x, y, 2) * 0.0721750f),
        ((gamma_compand(x, y, 0) * 0.0193339f) +
            (gamma_compand(x, y, 1) * 0.1191920f) +
            (gamma_compand(x, y, 2) * 0.9503041f)) / 1.08883f);

    const float kEpsilon = 216.0f / 24389.0f;
    const float kKappa = 24389.0f / 27.0f;

    // XYZ to Lab conversion. Repeated computation of the same function, on each
    // channel. I can't seem to figure out how to get Halide to treat Tuples
    // with the same function, but not do a reduction.
    Halide::Func lab("lab");
    lab(x, y) = Halide::Tuple(
        Halide::select(d65_white(x, y)[0] > kEpsilon,
            pow(d65_white(x, y)[0], 1.0f / 3.0f),
            ((kKappa * d65_white(x, y)[0]) + 16.0f) / 116.0f),
        Halide::select(d65_white(x, y)[1] > kEpsilon,
            pow(d65_white(x, y)[1], 1.0f / 3.0f),
            ((kKappa * d65_white(x, y)[1]) + 16.0f) / 116.0f),
        Halide::select(d65_white(x, y)[2] > kEpsilon,
            pow(d65_white(x, y)[2], 1.0f / 3.0f),
            ((kKappa * d65_white(x, y)[2]) + 16.0f) / 116.0f));

    Halide::Func xyz_to_lab("xyz_to_lab");
    xyz_to_lab(x, y) = Halide::Tuple(
        (116.0f * lab(x, y)[1]) - 16.0f,
        500.0f * (lab(x, y)[0] - lab(x, y)[1]),
        200.0f * (lab(x, y)[1] - lab(x, y)[2]));

    lab_output(x, y, c) = Halide::select(c == 0, xyz_to_lab(x, y)[0],
        Halide::select(c == 1, xyz_to_lab(x, y)[1],
            xyz_to_lab(x, y)[2]));
  }
};

HALIDE_REGISTER_GENERATOR(RgbToLab, rgb_to_lab)
