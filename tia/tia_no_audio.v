`ifndef TIA_TIA_NO_AUDIO_V
`define TIA_TIA_NO_AUDIO_V

`include "tia_ball_position_counter.v"
`include "tia_biphase_clock.v"
`include "tia_color_lum_registers.v"
`include "tia_divide_by_three.v"
`include "tia_horizontal_lfsr.v"
`include "tia_horizontal_lfsr_decoder.v"
`include "tia_horizontal_timing.v"
`include "tia_l.v"
`include "tia_motion_registers.v"
`include "tia_player_graphics.v"
`include "tia_playfield_registers.v"
`include "tia_write_address_decodes.v"

// One whole TIA chip, minus the audio support.

module tia_no_audio(
    // input
    d,     // 7:0 data
    a,     // 5:0 address
    osc,   // clock
    phi2,  // i/o clock drives latches on d and a
    rw,    // read, w_bar
    // output
    blk_bar,
    l,      // 2:0 luminosity
    c,      // 3:0 color
    syn,
    rdy,
    phi_theta  // clock divided by 3, for CPU
);

  input[7:0] d;
  input[5:0] a;
  input osc, phi2, rw;
  output blk_bar;
  output[2:0] l;
  output[3:0] c;
  output syn, rdy, phi_theta;

  wire[7:0] d;
  wire[5:0] a;
  wire osc, phi2, rw;
  wire blk_bar;
  wire[2:0] l;
  wire[3:0] c;
  wire syn, rdy, phi_theta;

  // === Page 2

  wire phi2_bar;
  assign phi2_bar = ~phi2;
  wire d0, d1, d2, d3, d4, d5, d6, d7;

  tia_l ld0(.in(d[0]), .follow(phi2), .latch(phi2_bar), .out(d0));
  tia_l ld1(.in(d[1]), .follow(phi2), .latch(phi2_bar), .out(d1));
  tia_l ld2(.in(d[2]), .follow(phi2), .latch(phi2_bar), .out(d2));
  tia_l ld3(.in(d[3]), .follow(phi2), .latch(phi2_bar), .out(d3));
  tia_l ld4(.in(d[4]), .follow(phi2), .latch(phi2_bar), .out(d4));
  tia_l ld5(.in(d[5]), .follow(phi2), .latch(phi2_bar), .out(d5));
  tia_l ld6(.in(d[6]), .follow(phi2), .latch(phi2_bar), .out(d6));
  tia_l ld7(.in(d[7]), .follow(phi2), .latch(phi2_bar), .out(d7));

  wire[5:0] al;
  tia_l la0(.in(a[0]), .follow(phi2), .latch(phi2_bar), .out(al[0]));
  tia_l la1(.in(a[1]), .follow(phi2), .latch(phi2_bar), .out(al[1]));
  tia_l la2(.in(a[2]), .follow(phi2), .latch(phi2_bar), .out(al[2]));
  tia_l la3(.in(a[3]), .follow(phi2), .latch(phi2_bar), .out(al[3]));
  tia_l la4(.in(a[4]), .follow(phi2), .latch(phi2_bar), .out(al[4]));
  tia_l la5(.in(a[5]), .follow(phi2), .latch(phi2_bar), .out(al[5]));

  wire vsyn, vblk, wsyn, rsyn, nsz0, nsz1, p0ci, p1ci, pfci, bkci, pfct, p0rf,
      p1rf, pf0, pf1, pf2, p0re, p1re, m0re, m1re, blre, auc0, auc1, auf0, auf1,
      auv0, auv1, p0gr, p1gr, m0en, m1en, blen, p0hm, p1hm, m0hm, m1hm, blhm,
      p0vd, p1vd, blvd, m0pre, m1pre, hmove, hmclr, cxclr;

  tia_write_address_decodes write_address_decodes(
      .a(al),
      .phi2(phi2),
      .w_bar(rw),
      .vsyn(vsyn),
      .vblk(vblk),
      .wsyn(wsyn),
      .rsyn(rsyn),
      .nsz0(nsz0),
      .nsz1(nsz1),
      .p0ci(p0ci),
      .p1ci(p1ci),
      .pfci(pfci),
      .bkci(bkci),
      .pfct(pfct),
      .p0rf(p0rf),
      .p1rf(p1rf),
      .pf0(pf0),
      .pf1(pf1),
      .pf2(pf2),
      .p0re(p0re),
      .p1re(p1re),
      .m0re(m0re),
      .m1re(m1re),
      .blre(blre),
      .auc0(auc0),
      .auc1(auc1),
      .auf0(auf0),
      .auf1(auf1),
      .auv0(auv0),
      .auv1(auv1),
      .p0gr(p0gr),
      .p1gr(p1gr),
      .m0en(m0en),
      .m1en(m1en),
      .blen(blen),
      .p0hm(p0hm),
      .p1hm(p1hm),
      .m0hm(m0hm),
      .m1hm(m1hm),
      .blhm(blhm),
      .p0vd(p0vd),
      .p1vd(p1vd),
      .blvd(blvd),
      .m0pre(m0pre),
      .m1pre(m1pre),
      .hmove(hmove),
      .hmclr(hmclr),
      .cxclr(cxclr));

  // === Page 1

  wire clk, clkp;
  assign clk = osc;

  assign clkp = ~clk;

  wire resphi0;
  tia_divide_by_three divide_by_three(.clk(clk),
                                      .resphi0(resphi0),
                                      .phi_theta(phi_theta));

  wire rsyn_gated;
  assign rsyn_gated = (~phi_theta) & rsyn;
  wire hphi1, hphi2, rsynl;
  tia_biphase_clock biphase_clock(.clk(clk),
                                  .r(rsyn_gated),
                                  .phi1(hphi1),
                                  .phi2(hphi2),
                                  .rl(rsynl));

  wire[5:0] horizontal_lfsr_out;
  wire shb, rsynd;
  tia_horizontal_lfsr horizontal_lfsr(.hphi1(hphi1),
                                      .hphi2(hphi2),
                                      .rsynl(rsynl),
                                      .out(horizontal_lfsr_out),
                                      .shb(shb),
                                      .rsynd(rsynd));

  wire rhs, cnt, rcb, shs, lrhb, rhb;
  tia_horizontal_lfsr_decoder horizontal_lfsr_decoder(.in(horizontal_lfsr_out),
                                                      .rhs(rhs),
                                                      .cnt(cnt),
                                                      .rcb(rcb),
                                                      .shs(shs),
                                                      .lrhb(lrhb),
                                                      .rhb(rhb));

  wire ref_bar, cntd, pf;
  tia_playfield_registers playfield_registers(.cnt(cnt),
                                              .rhb(rhb),
                                              .hphi1(hphi1),
                                              .hphi2(hphi2),
                                              .d0(d0),
                                              .d1(d1),
                                              .d2(d2),
                                              .d3(d3),
                                              .d4(d4),
                                              .d5(d5),
                                              .d6(d6),
                                              .d7(d7),
                                              .pf0(pf0),
                                              .pf1(pf1),
                                              .pf2(pf2),
                                              .clkp(clkp),
                                              .ref_bar(ref_bar),
                                              .cntd(cntd),
                                              .pf(pf));

  wire sec;
  wire p0ec_bar, p1ec_bar, m0ec_bar, m1ec_bar, blec_bar;

  tia_motion_registers motion_registers(.d4(d4),
                                        .d5(d5),
                                        .d6(d6),
                                        .d7(d7),
                                        .hmclr(hmclr),
                                        .p0hm(p0hm),
                                        .p1hm(p1hm),
                                        .m0hm(m0hm),
                                        .m1hm(m1hm),
                                        .blhm(blhm),
                                        .sec(sec),
                                        .hphi1(hphi1),
                                        .hphi2(hphi2));

  wire aphi1, aphi2, cb, vb, vb_bar, blank, motck;
  tia_horizontal_timing horizontal_timing(.hphi1(hphi1),
                                          .hphi2(hphi2),
                                          .rhs(rhs),
                                          .cnt(cnt),
                                          .rcb(rcb),
                                          .lrhb(lrhb),
                                          .rhb(rhb),
                                          .shs(shs),
                                          .vsyn(vsyn),
                                          .vblk(vblk),
                                          .rsynd(rsynd),
                                          .hmove(hmove),
                                          .clkp(clkp),
                                          .wsyn(wsyn),
                                          .d1(d1),
                                          .clk(clk),
                                          .shb(shb),
                                          .resphi0(resphi0),
                                          .aphi1(aphi1),
                                          .aphi2(aphi2),
                                          .cb(cb),
                                          .vb(vb),
                                          .vb_bar(vb_bar),
                                          .blank(blank),
                                          .motck(motck),
                                          .rdy(rdy),
                                          .sec(sec),
                                          .syn(syn));

  // === Pages 3 and 4

  wire p0, m0;
  tia_player_graphics player_graphics_0(.motck(motck),
                                        .pxec_bar(p0ec_bar),
                                        .pxre(p0re),
                                        .clkp(clkp),
                                        .mxre(m0re),
                                        .mxpre(m0pre),
                                        .pxvd(p0vd),
                                        .mxen(m0en),
                                        .nszx(nsz0),
                                        .mxec_bar(m0ec_bar),
                                        .pxgr(p0gr),
                                        .pygr(p1gr),
                                        .d0(d0),
                                        .d1(d1),
                                        .d2(d2),
                                        .d3(d3),
                                        .d4(d4),
                                        .d5(d5),
                                        .d6(d6),
                                        .d7(d7),
                                        .px(p0),
                                        .mx(m0));

  wire p1, m1;
  tia_player_graphics player_graphics_1(.motck(motck),
                                        .pxec_bar(p1ec_bar),
                                        .pxre(p1re),
                                        .clkp(clkp),
                                        .mxre(m1re),
                                        .mxpre(m1pre),
                                        .pxvd(p1vd),
                                        .mxen(m1en),
                                        .nszx(nsz1),
                                        .mxec_bar(m1ec_bar),
                                        .pxgr(p1gr),
                                        .pygr(p0gr),
                                        .d0(d0),
                                        .d1(d1),
                                        .d2(d2),
                                        .d3(d3),
                                        .d4(d4),
                                        .d5(d5),
                                        .d6(d6),
                                        .d7(d7),
                                        .px(p1),
                                        .mx(m1));

  // === Page 5

  wire pfct_bar;
  assign pfct_bar = ~pfct;
  wire score_bar, pfp_bar, blsiz1_bar, blsiz2, latch_bar, dump;
  tia_l pfctd0l(.in(d0), .follow(pfct), .latch(pfct_bar), .out_bar(ref_bar));
  tia_l pfctd1l(.in(d1), .follow(pfct), .latch(pfct_bar), .out_bar(score_bar));
  tia_l pfctd2l(.in(d2), .follow(pfct), .latch(pfct_bar), .out_bar(pfp_bar));
  tia_l pfctd4l(.in(d4), .follow(pfct), .latch(pfct_bar), .out_bar(blsiz1_bar));
  tia_l pfctd5l(.in(d5), .follow(pfct), .latch(pfct_bar), .out(blsiz2));

  wire vblk_bar;
  assign vblk_bar = ~vblk;
  tia_l vblkd6l(.in(d6), .follow(vblk), .latch(vblk_bar), .out_bar(latch_bar));
  tia_l vblkd7l(.in(d7), .follow(vblk), .latch(vblk_bar), .out(dump));

  wire bl;
  tia_ball_position_counter ball_position_counter(.motck(motck),
                                                  .blec_bar(blec_bar),
                                                  .blre(blre),
                                                  .blen(blen),
                                                  .p1gr(p1gr),
                                                  .blvd(blvd),
                                                  .blsiz2(blsiz2),
                                                  .blsiz1_bar(blsiz1_bar),
                                                  .clkp(clkp),
                                                  .d1(d1),
                                                  .d0(d0),
                                                  .bl(bl));

  wire l0, l1, l2, c0, c1, c2, c3;
  tia_color_lum_registers color_lum_registers(.p0(p0),
                                              .m0(m0),
                                              .p1(p1),
                                              .m1(m1),
                                              .pf(pf),
                                              .bl(bl),
                                              .blank(blank),
                                              .cntd(cntd),
                                              .score_bar(score_bar),
                                              .pfp_bar(pfp_bar),
                                              .d1(d1),
                                              .d2(d2),
                                              .d3(d3),
                                              .d4(d4),
                                              .d5(d5),
                                              .d6(d6),
                                              .d7(d7),
                                              .bkci(bkci),
                                              .pfci(pfci),
                                              .p1ci(p1ci),
                                              .p0ci(p0ci),
                                              .clkp(clkp),
                                              .blk_bar(blk_bar),
                                              .l0(l0),
                                              .l1(l1),
                                              .l2(l2),
                                              .c0(c0),
                                              .c1(c1),
                                              .c2(c2),
                                              .c3(c3));

  assign l[0] = l0;
  assign l[1] = l1;
  assign l[2] = l2;
  assign c[0] = c0;
  assign c[1] = c1;
  assign c[2] = c2;
  assign c[3] = c3;
endmodule  // tia_no_audio

`endif  // TIA_TIA_NO_AUDIO_V
