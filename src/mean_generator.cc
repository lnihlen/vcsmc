#include "Halide.h"

// Gaussian weighted mean on L*a*b* input, using only the Luminance channel.
// Size of kernel is 15 pixels square, so 7 px on each side around center pixel.
// With thanks to http://dev.theomader.com/gaussian-kernel-calculator/, with
// Sigma = 3.4 and Kernel Size of 15.
class Mean : public Halide::Generator<Mean> {
 public:
  Input<Halide::Buffer<float>> lab_input{"lab_input", 3};

  Output<Halide::Buffer<float>> mean_output{"mean_output", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};

    Func mean_x("mean_x");
    mean_x(x, y) =
        (0.014659f * lab_input(x - 7, y, 0)) +
        (0.025618f * lab_input(x - 6, y, 0)) +
        (0.041086f * lab_input(x - 5, y, 0)) +
        (0.060470f * lab_input(x - 4, y, 0)) +
        (0.081674f * lab_input(x - 3, y, 0)) +
        (0.101235f * lab_input(x - 2, y, 0)) +
        (0.115155f * lab_input(x - 1, y, 0)) +
        (0.120207f * lab_input(x, y, 0)) +
        (0.115155f * lab_input(x + 1, y, 0)) +
        (0.101235f * lab_input(x + 2, y, 0)) +
        (0.081674f * lab_input(x + 3, y, 0)) +
        (0.060470f * lab_input(x + 4, y, 0)) +
        (0.041086f * lab_input(x + 5, y, 0)) +
        (0.025618f * lab_input(x + 6, y, 0)) +
        (0.014659f * lab_input(x + 7, y, 0));

    Func mean_y("mean_y");
    mean_y(x, y) =
        (0.014659f * mean_x(x, y - 7)) +
        (0.025618f * mean_x(x, y - 6)) +
        (0.041086f * mean_x(x, y - 5)) +
        (0.060470f * mean_x(x, y - 4)) +
        (0.081674f * mean_x(x, y - 3)) +
        (0.101235f * mean_x(x, y - 2)) +
        (0.115155f * mean_x(x, y - 1)) +
        (0.120207f * mean_x(x, y)) +
        (0.115155f * mean_x(x, y + 1)) +
        (0.101235f * mean_x(x, y + 2)) +
        (0.081674f * mean_x(x, y + 3)) +
        (0.060470f * mean_x(x, y + 4)) +
        (0.041086f * mean_x(x, y + 5)) +
        (0.025618f * mean_x(x, y + 6)) +
        (0.014659f * mean_x(x, y + 7));

   mean_output(x, y) = mean_y(x, y);
  }
};

HALIDE_REGISTER_GENERATOR(Mean, mean)
