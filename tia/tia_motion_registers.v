`ifndef TIA_TIA_MOTION_REGISTERS_V
`define TIA_TIA_MOTION_REGISTERS_V

`include "tia_d1.v"
`include "tia_d2.v"
`include "tia_dl.v"

module tia_motion_registers_cell_a(hphi1, hphi2, sec, in, out);
  input hphi1, hphi2, sec, in;
  output out;
  wire hphi1, hphi2, sec, in;
  wire out;
  wire dlr;
  tia_d1 cell_a_d1(.in(in), .s1(hphi1), .s2(hphi2), .out(dlr));
  wire dlo;
  tia_dl cell_a_dl(.in(sec), .s1(hphi1), .s2(hphi2), .r(dlr), .out(dlo));
  assign out = ~(hphi1 & dlo);
endmodule  // tia_motion_registers_cell_a

// Cell B is the most numerous cell at 15 places. It stores an individual bit
// D4-D6 of the motion counter. They are wired vertically in a wired AND
// configuration, here on |waout|.
module tia_motion_registers_cell_b(d, follow, latch, hmclr, rin, waout, rout);
  input d, follow, latch, hmclr, rin;
  output waout, rout;
  wire d, follow, latch, hmclr, rin;
  wire waout, rout;
  reg mid;
  initial begin
    mid = 1;
  end
  always @(follow or d) begin
    if (follow) mid <= ~d;
  end
  wire latch_out;
  assign #1 latch_out = ~(hmclr | mid);
  always @(latch or latch_out) begin
    if (latch) mid <= ~latch_out;
  end
  assign rout = ~rin;
  assign waout = ~(mid & rout) & ~(latch_out & rin);
endmodule  // tia_motion_registers_cell_b

module tia_motion_registers_cell_c(d, follow, latch, hmclr, rin, waout, rout);
  input d, follow, latch, hmclr, rin;
  output waout, rout;
  wire d, follow, latch, hmclr, rin;
  wire waout, rout;
  reg mid;
  initial begin
    mid = 1;
  end
  always @(follow or d) begin
    if (follow) mid <= ~d;
  end
  wire latch_out;
  assign #1 latch_out = ~(hmclr | mid);
  always @(latch or latch_out) begin
    if (latch) mid <= ~latch_out;
  end
  assign rout = ~rin;
  assign waout = ~(mid & rin) & ~(latch_out & rout);
endmodule  // tia_motion_registers_cell_c

module tia_motion_registers_cell_d(
    lin, hphi1, hphi2, topin, waout, lout, botout);
  input lin, hphi1, hphi2, topin;
  output waout, lout, botout;
  wire lin, hphi1, hphi2, topin;
  wire waout, lout, botout;
  wire d2_in1;
  assign d2_in1 = ~(lin | topin);
  assign botout = ~((~topin) | lout);
  tia_d2 d2(.in1(d2_in1), .in2(botout), .s1(hphi1), .s2(hphi2), .out(lout));
  assign waout = ~lin;
endmodule  // tia_motion_registers_cell_d

module tia_motion_registers(
  // inputs
  d4,
  d5,
  d6,
  d7,
  hmclr,
  p0hm,
  p1hm,
  m0hm,
  m1hm,
  blhm,
  sec,
  hphi1,
  hphi2,
  // outputs
  p0ec_bar,
  p1ec_bar,
  m0ec_bar,
  m1ec_bar,
  blec_bar);

input d4, d5, d6, d7, hmclr, p0hm, p1hm, m0hm, m1hm, blhm, sec, hphi1, hphi2;
output p0ec_bar, p1ec_bar, m0ec_bar, m1ec_bar, blec_bar;

wire d4, d5, d6, d7, hmclr, p0hm, p1hm, m0hm, m1hm, blhm, sec, hphi1, hphi2;
wire p0ec_bar, p1ec_bar, m0ec_bar, m1ec_bar, blec_bar;

wire p0hm_bar;
assign p0hm_bar = ~p0hm;
wire p1hm_bar;
assign p1hm_bar = ~p1hm;
wire m0hm_bar;
assign m0hm_bar = ~m0hm;
wire m1hm_bar;
assign m1hm_bar = ~m1hm;
wire blhm_bar;
assign blhm_bar = ~blhm;

wire d4_rin, d5_rin, d6_rin, d7_rin;

// p0hm
wire p0hm_wand_d7, p0hm_wand_d6, p0hm_wand_d5, p0hm_wand_d4;
tia_motion_registers_cell_c p0hm_d7(.d(d7), .follow(p0hm), .latch(p0hm_bar),
    .hmclr(hmclr), .rin(d7_rin), .waout(p0hm_wand_d7));
tia_motion_registers_cell_b p0hm_d6(.d(d6), .follow(p0hm), .latch(p0hm_bar),
    .hmclr(hmclr), .rin(d6_rin), .waout(p0hm_wand_d6));
tia_motion_registers_cell_b p0hm_d5(.d(d5), .follow(p0hm), .latch(p0hm_bar),
    .hmclr(hmclr), .rin(d5_rin), .waout(p0hm_wand_d5));
tia_motion_registers_cell_b p0hm_d4(.d(d4), .follow(p0hm), .latch(p0hm_bar),
    .hmclr(hmclr), .rin(d4_rin), .waout(p0hm_wand_d4));
wire p0hm_wand;
assign p0hm_wand = p0hm_wand_d7 & p0hm_wand_d6 & p0hm_wand_d5 & p0hm_wand_d4;
tia_motion_registers_cell_a p0hm_ca(.hphi1(hphi1), .hphi2(hphi2), .sec(sec),
    .in(p0hm_wand), .out(p0ec_bar));

// p1hm
wire p1hm_wand_d7, p1hm_wand_d6, p1hm_wand_d5, p1hm_wand_d4;
tia_motion_registers_cell_c p1hm_d7(.d(d7), .follow(p1hm), .latch(p1hm_bar),
    .hmclr(hmclr), .rin(d7_rin), .waout(p1hm_wand_d7));
tia_motion_registers_cell_b p1hm_d6(.d(d6), .follow(p1hm), .latch(p1hm_bar),
    .hmclr(hmclr), .rin(d6_rin), .waout(p1hm_wand_d6));
tia_motion_registers_cell_b p1hm_d5(.d(d5), .follow(p1hm), .latch(p1hm_bar),
    .hmclr(hmclr), .rin(d5_rin), .waout(p1hm_wand_d5));
tia_motion_registers_cell_b p1hm_d4(.d(d4), .follow(p1hm), .latch(p1hm_bar),
    .hmclr(hmclr), .rin(d4_rin), .waout(p1hm_wand_d4));
wire p1hm_wand;
assign p1hm_wand = p1hm_wand_d7 & p1hm_wand_d6 & p1hm_wand_d5 & p1hm_wand_d4;
tia_motion_registers_cell_a p1hm_ca(.hphi1(hphi1), .hphi2(hphi2), .sec(sec),
    .in(p1hm_wand), .out(p1ec_bar));

// m0hm
wire m0hm_wand_d7, m0hm_wand_d6, m0hm_wand_d5, m0hm_wand_d4;
tia_motion_registers_cell_c m0hm_d7(.d(d7), .follow(m0hm), .latch(m0hm_bar),
    .hmclr(hmclr), .rin(d7_rin), .waout(m0hm_wand_d7));
tia_motion_registers_cell_b m0hm_d6(.d(d6), .follow(m0hm), .latch(m0hm_bar),
    .hmclr(hmclr), .rin(d6_rin), .waout(m0hm_wand_d6));
tia_motion_registers_cell_b m0hm_d5(.d(d5), .follow(m0hm), .latch(m0hm_bar),
    .hmclr(hmclr), .rin(d5_rin), .waout(m0hm_wand_d5));
tia_motion_registers_cell_b m0hm_d4(.d(d4), .follow(m0hm), .latch(m0hm_bar),
    .hmclr(hmclr), .rin(d4_rin), .waout(m0hm_wand_d4));
wire m0hm_wand;
assign m0hm_wand = m0hm_wand_d7 & m0hm_wand_d6 & m0hm_wand_d5 & m0hm_wand_d4;
tia_motion_registers_cell_a m0hm_ca(.hphi1(hphi1), .hphi2(hphi2), .sec(sec),
    .in(m0hm_wand), .out(m0ec_bar));

// m1hm
wire m1hm_wand_d7, m1hm_wand_d6, m1hm_wand_d5, m1hm_wand_d4;
tia_motion_registers_cell_c m1hm_d7(.d(d7), .follow(m1hm), .latch(m1hm_bar),
    .hmclr(hmclr), .rin(d7_rin), .waout(m1hm_wand_d7));
tia_motion_registers_cell_b m1hm_d6(.d(d6), .follow(m1hm), .latch(m1hm_bar),
    .hmclr(hmclr), .rin(d6_rin), .waout(m1hm_wand_d6));
tia_motion_registers_cell_b m1hm_d5(.d(d5), .follow(m1hm), .latch(m1hm_bar),
    .hmclr(hmclr), .rin(d5_rin), .waout(m1hm_wand_d5));
tia_motion_registers_cell_b m1hm_d4(.d(d4), .follow(m1hm), .latch(m1hm_bar),
    .hmclr(hmclr), .rin(d4_rin), .waout(m1hm_wand_d4));
wire m1hm_wand;
assign m1hm_wand = m1hm_wand_d7 & m1hm_wand_d6 & m1hm_wand_d5 & m1hm_wand_d4;
tia_motion_registers_cell_a m1hm_ca(.hphi1(hphi1), .hphi2(hphi2), .sec(sec),
    .in(m1hm_wand), .out(m1ec_bar));

wire d7_lin, d6_lin, d5_lin, d4_lin;

// blhm
wire blhm_wand_d7, blhm_wand_d6, blhm_wand_d5, blhm_wand_d4;
tia_motion_registers_cell_c blhm_d7(.d(d7), .follow(blhm), .latch(blhm_bar),
    .hmclr(hmclr), .rin(d7_rin), .waout(blhm_wand_d7), .rout(d7_lin));
tia_motion_registers_cell_b blhm_d6(.d(d6), .follow(blhm), .latch(blhm_bar),
    .hmclr(hmclr), .rin(d6_rin), .waout(blhm_wand_d6), .rout(d6_lin));
tia_motion_registers_cell_b blhm_d5(.d(d5), .follow(blhm), .latch(blhm_bar),
    .hmclr(hmclr), .rin(d5_rin), .waout(blhm_wand_d5), .rout(d5_lin));
tia_motion_registers_cell_b blhm_d4(.d(d4), .follow(blhm), .latch(blhm_bar),
    .hmclr(hmclr), .rin(d4_rin), .waout(blhm_wand_d4), .rout(d4_lin));
wire blhm_wand;
assign blhm_wand = blhm_wand_d7 & blhm_wand_d6 & blhm_wand_d5 & blhm_wand_d4;
tia_motion_registers_cell_a blhm_ca(.hphi1(hphi1), .hphi2(hphi2), .sec(sec),
    .in(blhm_wand), .out(blec_bar));

// ripple counters
wire d6_botout, d5_botout, d4_botout;
wire d7_wand, d6_wand, d5_wand, d4_wand;
wire ripple_and = d7_wand & d6_wand & d5_wand & d4_wand;
wire d4_topin;
assign d4_topin = ~((~sec) & ripple_and);
tia_motion_registers_cell_d d7_ripple(.lin(d7_lin), .hphi1(hphi1),
    .hphi2(hphi2), .topin(d6_botout), .waout(d7_wand), .lout(d7_rin));
tia_motion_registers_cell_d d6_ripple(.lin(d6_lin), .hphi1(hphi1),
    .hphi2(hphi2), .topin(d5_botout), .waout(d6_wand), .lout(d6_rin),
    .botout(d6_botout));
tia_motion_registers_cell_d d5_ripple(.lin(d5_lin), .hphi1(hphi1),
    .hphi2(hphi2), .topin(d4_botout), .waout(d5_wand), .lout(d5_rin),
    .botout(d5_botout));
tia_motion_registers_cell_d d4_ripple(.lin(d4_lin), .hphi1(hphi1),
    .hphi2(hphi2), .topin(d4_topin), .waout(d4_wand), .lout(d4_rin),
    .botout(d4_botout));

endmodule  // tia_motion_registers

`endif  // TIA_TIA_MOTION_REGISTERS
