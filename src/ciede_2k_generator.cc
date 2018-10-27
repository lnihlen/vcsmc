#include "Halide.h"

#include "constants.h"

class Ciede2k : public Halide::Generator<Ciede2k> {
 public:
  Input<Halide::Buffer<float>> lab_a_input{"lab_a_input", 3};
  Input<Halide::Buffer<float>> lab_b_input{"lab_b_input", 3};

  Output<Halide::Buffer<float>> ciede_output{"ciede_output", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    Halide::Func c_star_mean_pow("C_star_mean_pow");
    C_star_mean_pow(x, y) = pow(
      (sqrt((lab_a_input(x, y, 1) * lab_a_input(x, y, 1)) +
            (lab_a_input(x, y, 2) * lab_a_input(x, y, 2))) +
       sqrt((lab_b_input(x, y, 1) * lab_b_input(x, y, 1)) +
            (lab_b_input(x, y, 2) * lab_b_input(x, y, 2)))) / 2.0f, 7.0f);

    Halide::Func G("G");
    G(x, y) = 0.5f * (1.0f - sqrt(C_star_mean_pow(x, y) /
        (C_star_mean_pow(x, y) + 6103515625.0)));  // 25^7

    Halide::Func a("a");
    a(x, y) = Halide::Tuple(
        (1.0f + G(x, y)) * lab_a_input(x, y, 1),
        (1.0f + G(x, y)) * lab_b_input(x, y, 1));

    Halide::Func C("C");

  }
};

HALIDE_REGISTER_GENERATOR(Ciede2k, ciede2k)
