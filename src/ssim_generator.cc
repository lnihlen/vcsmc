#include "Halide.h"

#include "constants.h"

class Ssim : public Halide::Generator<Ssim> {
 public:
  Input<Halide::Buffer<float>> mean_in_1{"mean_in_1", 2};
  Input<Halide::Buffer<float>> variance_in_1{"variance_in_1", 2};
  Input<Halide::Buffer<float>> mean_in_2{"mean_in_2", 2};
  Input<Halide::Buffer<float>> variance_in_2{"variance_in_2", 2};
  Input<Halide::Buffer<float>> covariance_in{"covariance_in", 2};

  Output<Halide::Buffer<float>> ssim_out{"ssim_out", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};

    ssim_out(x, y) =
        (((2.0f * mean_in_1(x, y) * mean_in_2(x, y)) + vcsmc::kSSIMC1) *
         ((2.0f * covariance_in(x, y)) + vcsmc::kSSIMC2)) /
        (((mean_in_1(x, y) * mean_in_1(x, y)) +
          (mean_in_2(x, y) * mean_in_2(x, y)) + vcsmc::kSSIMC1) *
         (variance_in_1(x, y) + variance_in_2(x, y) + vcsmc::kSSIMC2));
  }
};

HALIDE_REGISTER_GENERATOR(Ssim, ssim)
