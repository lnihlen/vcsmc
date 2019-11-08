#include "Halide.h"

// Gaussian weighted mean on L*a*b* input, using only the Luminance channel.
// Size of kernel is 15 pixels square, so 7 px on each side around center pixel.
// With thanks to http://dev.theomader.com/gaussian-kernel-calculator/, with
// Sigma = 3.4 and Kernel Size of 15.
class Mean : public Halide::Generator<Mean> {
 public:
  Input<Halide::Buffer<float>> lab_in{"lab_in", 3};

  Output<Halide::Buffer<float>> mean_out{"mean_out", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};

    Func clamp("clamp");
    clamp = Halide::BoundaryConditions::constant_exterior(lab_in, 0);

    Func mean_x("mean_x");
    mean_x(x, y) =
        (0.014659f * clamp(x - 7, y, 0)) +
        (0.025618f * clamp(x - 6, y, 0)) +
        (0.041086f * clamp(x - 5, y, 0)) +
        (0.060470f * clamp(x - 4, y, 0)) +
        (0.081674f * clamp(x - 3, y, 0)) +
        (0.101235f * clamp(x - 2, y, 0)) +
        (0.115155f * clamp(x - 1, y, 0)) +
        (0.120207f * clamp(x, y, 0)) +
        (0.115155f * clamp(x + 1, y, 0)) +
        (0.101235f * clamp(x + 2, y, 0)) +
        (0.081674f * clamp(x + 3, y, 0)) +
        (0.060470f * clamp(x + 4, y, 0)) +
        (0.041086f * clamp(x + 5, y, 0)) +
        (0.025618f * clamp(x + 6, y, 0)) +
        (0.014659f * clamp(x + 7, y, 0));

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

   mean_out(x, y) = mean_y(x, y);
  }
};

HALIDE_REGISTER_GENERATOR(Mean, mean)
