#include "Halide.h"

class Covariance : public Halide::Generator<Covariance> {
 public:
  Input<Halide::Buffer<float>> lab_in_1{"lab_in_1", 3};
  Input<Halide::Buffer<float>> mean_in_1{"mean_in_1", 2};
  Input<Halide::Buffer<float>> lab_in_2{"lab_in_2", 3};
  Input<Halide::Buffer<float>> mean_in_2{"mean_in_2", 2};
  Input<Halide::Buffer<float>> kernel_in{"kernel_in", 2};

  Output<Halide::Buffer<float>> covariance_out{"covariance_out", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};

    Func lab_clamp_1("lab_clamp_1");
    lab_clamp_1 = Halide::BoundaryConditions::constant_exterior(lab_in_1, 0);
    Func mean_clamp_1("mean_clamp_1");
    mean_clamp_1 = Halide::BoundaryConditions::constant_exterior(mean_in_1, 0);
    Func lab_clamp_2("lab_clamp_2");
    lab_clamp_2 = Halide::BoundaryConditions::constant_exterior(lab_in_2, 0);
    Func mean_clamp_2("mean_clamp_2");
    mean_clamp_2 = Halide::BoundaryConditions::constant_exterior(mean_in_2, 0);

    RDom r(-7, 15, -7, 15);
    Func covariance("covariance");
    covariance(x, y) = 0.0f;
    covariance(x, y) += kernel_in(r.x + 7, r.y + 7) *
        (lab_clamp_1(x + r.x, y + r.y, 0) - mean_clamp_1(x + r.x, y + r.y)) *
        (lab_clamp_2(x + r.x, y + r.y, 0) - mean_clamp_2(x + r.x, y + r.y));
    covariance_out(x, y) = covariance(x, y);
  }
};

HALIDE_REGISTER_GENERATOR(Covariance, covariance)
