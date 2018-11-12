#include "Halide.h"

class Variance : public Halide::Generator<Variance> {
 public:
  Input<Halide::Buffer<float>> lab_in{"lab_in", 3};
  Input<Halide::Buffer<float>> mean_in{"mean_in", 2};

  Output<Halide::Buffer<float>> variance_out{"variance_out", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};

    variance_out(x, y) = lab_in(x, y, 0) - mean_in(x, y);
  }
};

HALIDE_REGISTER_GENERATOR(Variance, variance)
