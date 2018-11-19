#include "Halide.h"

// Gaussian weighted standard deviation.
class Variance : public Halide::Generator<Variance> {
 public:
  Input<Halide::Buffer<float>> lab_in{"lab_in", 3};
  Input<Halide::Buffer<float>> mean_in{"mean_in", 2};
  Input<Halide::Buffer<float>> kernel_in{"kernel_in", 2};

  Output<Halide::Buffer<float>> variance_out{"variance_out", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};

    Func lab_clamp("lab_clamp");
    lab_clamp = Halide::BoundaryConditions::constant_exterior(lab_in, 0);
    Func mean_clamp("mean_clamp");
    mean_clamp = Halide::BoundaryConditions::constant_exterior(mean_in, 0);

    RDom r(-7, 15, -7, 15);
    Func variance("variance");
    variance(x, y) = 0.0f;
    variance(x, y) += kernel_in(r.x + 7, r.y + 7) *
        (lab_clamp(x + r.x, y + r.y, 0) - mean_clamp(x + r.x, y + r.y)) *
        (lab_clamp(x + r.x, y + r.y, 0) - mean_clamp(x + r.x, y + r.y));
    variance_out(x, y) = variance(x, y);
  }
};

HALIDE_REGISTER_GENERATOR(Variance, variance)
