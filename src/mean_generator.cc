#include "Halide.h"

// Gaussian weighted mean on L*a*b* input, using only the Luminance channel.
// Size of kernel is 7 pixels square, so 3 px on each side around center pixel.
class Mean : public Halide::Generator<Mean> {
 public:
  Input<Halide::Buffer<float>> lab_input{"lab_input", 3};

  Output<Halide::Buffer<float>> mean_output{"mean_output", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};
    mean_output(x, y) = lab_input(x, y, 0);
  }
};

HALIDE_REGISTER_GENERATOR(Mean, mean)
