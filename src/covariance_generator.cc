#include "Halide.h"

class Covariance : public Halide::Generator<Covariance> {
 public:
  Input<Halide::Buffer<float>> lab_input_1{"lab_input_1", 3};
  Input<Halide::Buffer<float>> mean_input_1{"mean_input_1", 2};
  Input<Halide::Buffer<float>> lab_input_2{"lab_input_2", 3};
  Input<Halide::Buffer<float>> mean_input_2{"mean_input_2", 2};

  Output<Halide::Buffer<float>> covariance_output{"covariance_output", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};
    covariance_output(x, y) = (lab_input_1(x, y, 0) - mean_input_1(x, y)) *
        (lab_input_2(x, y, 0) - mean_input_2(x, y));
  }
};

HALIDE_REGISTER_GENERATOR(Covariance, covariance)
