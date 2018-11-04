#include "Halide.h"

#include "constants.h"

class Ciede2k : public Halide::Generator<Ciede2k> {
 public:
  Input<Halide::Buffer<float>> lab_a_input{"lab_input_1", 3};
  Input<Halide::Buffer<float>> lab_b_input{"lab_input_2", 3};

  Output<Halide::Buffer<float>> ciede_output{"ciede_output", 2};

  void generate() {
    Halide::Var x{"x"}, y{"y"}, c{"c"};

    Halide::Func L_star_1("L_star_1");
    L_star_1(x, y) = lab_input_1(x, y, 0);
    Halide::Func a_star_1("a_star_1");
    a_star_1(x, y) = lab_input_1(x, y, 1);
    Halide::Func b_star_1("b_star_1");
    b_star_1(x, y) = lab_input_1(x, y, 2);

    Halide::Func L_star_2("L_star_2");
    L_star_2(x, y) = lab_input_2(x, y, 0);
    Halide::Func a_star_1("a_star_2");
    a_star_2(x, y) = lab_input_2(x, y, 1);
    Halide::Func b_star_1("b_star_2");
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
        (C_star_mean_pow(x, y) + 6103515625.0)));  // 25^7

    Halide::Func a_1("a_1");
    a_1(x, y) = (1.0 + G) * a_star_1(x, y);
    Halide::Func a_2("a_2");
    a_2(x, y) = (1.0 + G) * a_star_2(x, y);

    Halide::Func C_1("C_1");
    C_1(x, y) = sqrt((a_1(x, y) * a_1(x, y)) +
                     (b_star_1(x, y) * b_star_1(x, y)));
    Halide::Func C_2("C_2");
    C_2(x, y) = sqrt((a_2(x, y) * a_2(x, y)) +
                     (b_star_2(x, y) * b_star_2(x, y)));

    // Intermediate term h_1_atan not part of standard formulation.
    Halide::Func h_1_atan("h_1_atan");
    h_1_atan(x, y) = select(a_1(x, y) == 0.0 && b_star_1(x, y) == 0.0,
                            0.0,
                            atan2(b_star_1(x, y), a_1(x, y)));
    Halide::Func h_2_atan("h_2_atan");
    h_2_atan(x, y) = select(a_2(x, y) == 0.0 && b_star_2(x, y) == 0.0,
                            0.0,
                            atan2(b_star_2(x, y), a_2(x, y)));
    Halide::Func h_1("h_1");
    h_1(x, y) = select(h_1_atan(x, y) < 0.0,
                       (h_1_atan(x, y) + (2.0 * kPi)) * (180.0 / kPi),
                       h_1_atan(x, y) * (180.0 / kPi));
    Halide::Func h_2("h_2");
    h_2(x, y) = select(h_2_atan(x, y) < 0.0,
                       (h_2_atan(x, y) + (2.0 * kPi)) * (180.0 / kPi),
                       h_2_atan(x, y) * (180.0 / kPi));

    Halide::Func del_L("del_L");
    del_L(x, y) = L_star_2(x, y) - L_star_1(x, y);
    Halide::Func del_C("del_C");
    del_C(x, y) = C_2(x, y) - C_1(x, y);
    Halide::Func C_product("C_product");
    C_product(x, y) = C_1(x, y) * C_2(x, y);
    Halide::Func h_diff("h_diff");
    h_diff(x, y) = h_2(x, y) - h_1(x, y);
    Halide::Func del_h("del_h");
    del_h(x, y) = select(C_product(x, y) != 0.0,
                         select(h_diff(x, y) < -180.0,
                                h_diff(x, y) + 360.0,
                                select(h_diff(x, y) <= 180.0,
                                       h_diff(x, y),
                                       h_diff(x, y) - 360.0)),
                         0.0);

    Halide::Func del_H("del_H");
    del_H(x, y) = 2.0 * sqrt(C_product(x, y)) *
        sin((del_h(x, y) * (kPi / 180.0)) / 2.0);


    Halide::Func L_mean("L_mean");
    L_mean(x, y) = (L_star_1(x, y) + L_star_2(x, y)) / 2.0;
    Halide::Func C_mean("C_mean");
    C_mean(x, y) = (C_1(x, y) + C_2(x, y)) / 2.0;
    Halide::Func h_sum("h_sum");
    h_sum(x, y) = h_1(x, y) + h_2(x, y);
    Halide::Func h_abs("h_abs");
    h_abs(x, y) = abs(h_1(x, y) - h_2(x, y));
    Halide::Func h_mean("h_mean");
    h_mean(x, y) = select(C_product(x, y) != 0.0,
                          select(h_abs(x, y) <= 180.0,
                                 h_sum(x, y) / 2.0,
                                 select(h_sum(x, y) < 360.0,
                                        (h_sum(x, y) + 360.0) / 2.0,
                                        (h_sum(x, y) - 360.0) / 2.0)),
                          h_sum(x, y));

    Halide::Func T("T");
    T(x, y) = 1.0 - (0.17 * cos((h_mean(x, y) - 30.0) * (kPi / 180.0))) +
                    (0.24 * cos(2.0 * h_mean(x, y) * (kPi / 180.0))) +
                    (0.32 * cos(((3.0 * h_mean(x, y)) + 6.0) * (kPi / 180.0))) -
                    (0.20 * cos(((4.0 * h_mean(x, y)) - 63.0) * (kPi / 180.0)));

    Halide::Func del_theta("del_theta");
    del_theta(x, y) = 30.0 *
        exp(-1.0 * pow((h_mean(x, y) - 275.0) / 25.0, 2.0));

    Halide::Func C_mean_pow("C_mean_pow");
    C_mean_pow(x, y) = pow(C_mean(x, y), 7.0);

    Halide::Func R_c("R_c");
    R_c(x, y) = 2.0 * sqrt(C_mean_pow(x, y) /
                          (C_mean_pow(x, y) + 6103515625.0));

    // Intermediate term L_int not part of standard formulation.
    Halide::Func L_int("L_int");
    L_int(x, y) = pow(L_mean(x, y) - 50.0, 2.0);
    Halide::Func S_L("S_L");
    S_L(x, y) = 1.0 + ((0.015 * L_int(x, y)) / sqrt(20.0 + L_int(x, y)));
    Halide::Func S_C("S_C");
    S_C(x, y) = 1.0 + (0.045 * C_mean(x, y));
    Halide::Func S_H("S_H");
    S_H(x, y) = 1.0 + (0.015 * C_mean(x, y) * T(x, y));
    Halide::Func R_T("R_T");
    R_T(x, y) = -1.0 * sin(2.0 * del_theta(x, y) * (kPi / 180.0)) * R_c(x, y);
    Halide::Func delta_e("delta_e");
    delta_e(x, y) = sqrt(pow(del_L(x, y) / S_L(x, y), 2.0) +
                         pow(del_C(x, y) / S_C(x, y), 2.0) +
                         pow(del_H(x, y) / S_H(x, y), 2.0) +
                         R_T(x, y) *
                         (del_C(x, y) / S_C(x, y)) *
                         (del_H(x, y) / S_C(x, y)));

    ciede_output(x, y) = delta_e(x, y);
  }
};

HALIDE_REGISTER_GENERATOR(Ciede2k, ciede2k)
