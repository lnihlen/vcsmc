`ifndef TIA_TIA_PLAYER_GRAPHICS_V
`define TIA_TIA_PLAYER_GRAPHICS_V

`include "tia_l.v"
`include "tia_missile_position_counter.v"
`include "tia_player_graphics_register.v"
`include "tia_player_graphics_scan_counter.v"
`include "tia_player_position_counter.v"

module tia_player_graphics(
    // input
    motck,
    pxec_bar,
    pxre,
    clkp,
    mxre,
    mxpre,
    pxvd,
    mxen,
    nszx,
    mxec_bar,
    pxgr,
    pygr,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    // output
    px,
    mx
);

  input motck, pxec_bar, pxre, clkp, mxre, mxpre, pxvd, mxen, nszx, mxec_bar,
      pxgr, pygr, d0, d1, d2, d3, d4, d5, d6, d7;
  output px, mx;

  wire motck, pxec_bar, pxre, clkp, mxre, mxpre, pxvd, mxen, nszx, mxec_bar,
      pxgr, pygr, d0, d1, d2, d3, d4, d5, d6, d7;
  wire px, mx;

  wire nszx_bar;
  assign nszx_bar = ~nszx;
  wire nz0_bar, nz1_bar, nz2_bar, nz4_bar, nz5_bar;
  tia_l nszxd0l(.in(d0), .follow(nszx), .latch(nszx_bar), .out_bar(nz0_bar));
  tia_l nszxd1l(.in(d1), .follow(nszx), .latch(nszx_bar), .out_bar(nz1_bar));
  tia_l nszxd2l(.in(d2), .follow(nszx), .latch(nszx_bar), .out_bar(nz2_bar));
  tia_l nszxd4l(.in(d4), .follow(nszx), .latch(nszx_bar), .out_bar(nz4_bar));
  tia_l nsxzd5l(.in(d5), .follow(nszx), .latch(nszx_bar), .out_bar(nz5_bar));

  wire mxen_bar;
  assign mxen_bar = ~mxen;
  wire missile_enable;
  tia_l mxend1l(.in(d1), .follow(mxen), .latch(mxen_bar), .out(missile_enable));

  wire pxvd_bar;
  assign pxvd_bar = ~pxvd;
  wire player_vert_delay_bar;
  tia_l pxvdd0l(.in(d0), .follow(pxvd), .latch(pxvd_bar),
      .out_bar(player_vert_delay_bar));

  wire mxpre_bar;
  assign mxpre_bar = ~mxpre;
  wire missile_to_player_reset_bar;
  tia_l mxpred1l(.in(d1), .follow(mxpre), .latch(mxpre_bar),
      .out_bar(missile_to_player_reset_bar));

  wire pxre_bar;
  assign pxre_bar = ~pxre;
  wire player_reflect_bar;
  tia_l pxred3l(.in(d3), .follow(pxre), .latch(pxre_bar),
      .out_bar(player_reflect_bar));

  wire start_bar, fstob, pck, pphi1, pphi2, count_bar;
  tia_player_position_counter player_position_counter(
      .motck(motck),
      .pec_bar(pxec_bar),
      .pre(pxre),
      .nz0_bar(nz0_bar),
      .nz1_bar(nz1_bar),
      .nz2_bar(nz2_bar),
      .start_bar(start_bar),
      .fstob(fstob),
      .pck(pck),
      .pphi1(pphi1),
      .pphi2(pphi2),
      .count_bar(count_bar));

  wire new, old, missile_to_player_reset, gs0, gs1, gs2;
  tia_player_graphics_scan_counter graphics_scan_counter(
      .start_bar(start_bar),
      .fstob(fstob),
      .pck(pck),
      .count_bar(count_bar),
      .new(new),
      .old(old),
      .player_vert_delay_bar(player_vert_delay_bar),
      .missile_to_player_reset_bar(missile_to_player_reset_bar),
      .player_reflect_bar(player_reflect_bar),
      .clkp(clkp),
      .missile_to_player_reset(missile_to_player_reset),
      .gs0(gs0),
      .gs1(gs1),
      .gs2(gs2),
      .p(px));

  wire missile_reset;
  tia_missile_position_counter missile_position_counter(
      .motck(motck),
      .mec_bar(mxec_bar),
      .clkp(clkp),
      .nz0_bar(nz0_bar),
      .nz1_bar(nz1_bar),
      .nz2_bar(nz2_bar),
      .missile_enable(missile_enable),
      .missile_to_player_reset_bar(missile_to_player_reset_bar),
      .nz4_bar(nz4_bar),
      .nz5_bar(nz5_bar),
      .missile_reset(missile_reset),
      .missile_to_player_reset(missile_to_player_reset),
      .m(mx));

  tia_player_graphics_register player_graphics_register(.d0(d0),
                                                        .d1(d1),
                                                        .d2(d2),
                                                        .d3(d3),
                                                        .d4(d4),
                                                        .d5(d5),
                                                        .d6(d6),
                                                        .d7(d7),
                                                        .gs0(gs0),
                                                        .gs1(gs1),
                                                        .gs2(gs2),
                                                        .p0gr(pxgr),
                                                        .p1gr(pygr),
                                                        .new(new),
                                                        .old(old));

endmodule  // tia_player_graphics

`endif  // TIA_TIA_PLAYER_GRAPHICS_V
