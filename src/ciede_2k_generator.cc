#include "Halide.h"

#include "constants.h"

class Ciede2k : public Halide::Generator<Ciede2k> {
 public:
  Input<Halide::Buffer<float>> lab_input_1{"lab_input_1", 3};
  Input<Halide::Buffer<float>> lab_input_2{"lab_input_2", 3};

  Output<Halide::Buffer<float>> ciede_output{"ciede_output", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"};

    Halide::Func L_star_1("L_star_1");
    L_star_1(x, y) = lab_input_1(x, y, 0);
    Halide::Func a_star_1("a_star_1");
    a_star_1(x, y) = lab_input_1(x, y, 1);
    Halide::Func b_star_1("b_star_1");
    b_star_1(x, y) = lab_input_1(x, y, 2);

    Halide::Func L_star_2("L_star_2");
    L_star_2(x, y) = lab_input_2(x, y, 0);
    Halide::Func a_star_2("a_star_2");
    a_star_2(x, y) = lab_input_2(x, y, 1);
    Halide::Func b_star_2("b_star_2");
    b_star_2(x, y) = lab_input_2(x, y, 2);

    Halide::Func C_star_1("C_star_1");
    C_star_1(x, y) = sqrt((a_star_1(x, y) * a_star_1(x, y)) +
                          (b_star_1(x, y) * b_star_1(x, y)));
    Halide::Func C_star_2("C_star_2");
    C_star_2(x, y) = sqrt((a_star_2(x, y) * a_star_2(x, y)) +
                          (b_star_2(x, y) * b_star_2(x, y)));

    Halide::Func C_star_mean_pow("C_star_mean_pow");
    C_star_mean_pow(x, y) = pow((C_star_1(x, y) + C_star_2(x, y)) / 2.0f, 7.0f);

    Halide::Func G("G");
    G(x, y) = 0.5f * (1.0f - sqrt(C_star_mean_pow(x, y) /
        (C_star_mean_pow(x, y) + 6103515625.0f)));  // 25^7

    Halide::Func a_1("a_1");
    a_1(x, y) = (1.0f + G(x, y)) * a_star_1(x, y);
    Halide::Func a_2("a_2");
    a_2(x, y) = (1.0f + G(x, y)) * a_star_2(x, y);

    Halide::Func C_1("C_1");
    C_1(x, y) = sqrt((a_1(x, y) * a_1(x, y)) +
                     (b_star_1(x, y) * b_star_1(x, y)));
    Halide::Func C_2("C_2");
    C_2(x, y) = sqrt((a_2(x, y) * a_2(x, y)) +
                     (b_star_2(x, y) * b_star_2(x, y)));

    // Intermediate term h_1_atan not part of standard formulation.
    Halide::Func h_1_atan("h_1_atan");
    h_1_atan(x, y) = select(a_1(x, y) == 0.0f && b_star_1(x, y) == 0.0f, 0.0f,
                            atan2(b_star_1(x, y), a_1(x, y)));
    Halide::Func h_2_atan("h_2_atan");
    h_2_atan(x, y) = select(a_2(x, y) == 0.0f && b_star_2(x, y) == 0.0f, 0.0f,
                            atan2(b_star_2(x, y), a_2(x, y)));
    Halide::Func h_1("h_1");
    h_1(x, y) = select(h_1_atan(x, y) < 0.0f,
                       (h_1_atan(x, y) + (2.0f * vcsmc::kPi)) *
                            (180.0f / vcsmc::kPi),
                       h_1_atan(x, y) * (180.0f / vcsmc::kPi));
    Halide::Func h_2("h_2");
    h_2(x, y) = select(h_2_atan(x, y) < 0.0f,
                       (h_2_atan(x, y) + (2.0f * vcsmc::kPi)) *
                            (180.0f / vcsmc::kPi),
                       h_2_atan(x, y) * (180.0f / vcsmc::kPi));

    Halide::Func del_L("del_L");
    del_L(x, y) = L_star_2(x, y) - L_star_1(x, y);

    Halide::Func del_C("del_C");
    del_C(x, y) = C_2(x, y) - C_1(x, y);

    Halide::Func C_product("C_product");
    C_product(x, y) = C_1(x, y) * C_2(x, y);

    Halide::Func h_diff("h_diff");
    h_diff(x, y) = h_2(x, y) - h_1(x, y);

    Halide::Func del_h("del_h");
    del_h(x, y) = select(C_product(x, y) == 0.0f, 0.0f,
                         h_diff(x, y) < -180.0f, h_diff(x, y) + 360.0f,
                         h_diff(x, y) <= 180.0f, h_diff(x, y),
                         h_diff(x, y) - 360.0f);

    Halide::Func del_H("del_H");
    del_H(x, y) = 2.0f * sqrt(C_product(x, y)) *
        sin((del_h(x, y) * (vcsmc::kPi / 180.0f)) / 2.0f);


    Halide::Func L_mean("L_mean");
    L_mean(x, y) = (L_star_1(x, y) + L_star_2(x, y)) / 2.0f;

    Halide::Func C_mean("C_mean");
    C_mean(x, y) = (C_1(x, y) + C_2(x, y)) / 2.0f;

    Halide::Func h_sum("h_sum");
    h_sum(x, y) = h_1(x, y) + h_2(x, y);

    Halide::Func h_abs("h_abs");
    h_abs(x, y) = abs(h_1(x, y) - h_2(x, y));

    Halide::Func h_mean("h_mean");
    h_mean(x, y) = select(C_product(x, y) == 0.0f, h_sum(x, y),
                          h_abs(x, y) <= 180.0f, h_sum(x, y) / 2.0f,
                          h_sum(x, y) < 360.0f, (h_sum(x, y) + 360.0f) / 2.0f,
                          (h_sum(x, y) - 360.0f) / 2.0f);

    Halide::Func T("T");
    T(x, y) = 1.0f - (0.17f * cos((h_mean(x, y) - 30.0f) *
                                  (vcsmc::kPi / 180.0f))) +
                     (0.24f * cos(2.0f * h_mean(x, y) *
                                  (vcsmc::kPi / 180.0f))) +
                     (0.32f * cos(((3.0f * h_mean(x, y)) + 6.0f) *
                                  (vcsmc::kPi / 180.0f))) -
                     (0.20f * cos(((4.0f * h_mean(x, y)) - 63.0f) *
                                  (vcsmc::kPi / 180.0f)));

    Halide::Func del_theta("del_theta");
    del_theta(x, y) = 30.0f *
        exp(-1.0f * pow((h_mean(x, y) - 275.0f) / 25.0f, 2.0f));

    Halide::Func C_mean_pow("C_mean_pow");
    C_mean_pow(x, y) = pow(C_mean(x, y), 7.0f);

    Halide::Func R_c("R_c");
    R_c(x, y) = 2.0f * sqrt(C_mean_pow(x, y) /
                           (C_mean_pow(x, y) + 6103515625.0f));

    // Intermediate term L_int not part of standard formulation.
    Halide::Func L_int("L_int");
    L_int(x, y) = pow(L_mean(x, y) - 50.0f, 2.0f);

    Halide::Func S_L("S_L");
    S_L(x, y) = 1.0f + ((0.015f * L_int(x, y)) / sqrt(20.0f + L_int(x, y)));

    Halide::Func S_C("S_C");
    S_C(x, y) = 1.0f + (0.045f * C_mean(x, y));

    Halide::Func S_H("S_H");
    S_H(x, y) = 1.0f + (0.015f * C_mean(x, y) * T(x, y));

    Halide::Func R_T("R_T");
    R_T(x, y) = -1.0f *
        sin(2.0f * del_theta(x, y) * (vcsmc::kPi / 180.0f)) * R_c(x, y);

    Halide::Func delta_e("delta_e");
    delta_e(x, y) = sqrt(pow(del_L(x, y) / S_L(x, y), 2.0f) +
                         pow(del_C(x, y) / S_C(x, y), 2.0f) +
                         pow(del_H(x, y) / S_H(x, y), 2.0f) +
                         (R_T(x, y) * (del_C(x, y) / S_C(x, y)) *
                            (del_H(x, y) / S_H(x, y))));

    ciede_output(x, y) = delta_e(x, y);
  }
};

HALIDE_REGISTER_GENERATOR(Ciede2k, ciede_2k)
