`include "sr.v"
`include "tia_d1.v"
`include "tia_f1.v"
`include "tia_l.v"
`include "tia_playfield_registers_cell.v"

module tia_playfield_registers(
    // inputs:
    cnt,
    rhb,
    hphi1,
    hphi2,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    pf0,
    pf1,
    pf2,
    clkp,
    ref_bar,
    // outputs:
    cntd,
    pf);

input cnt, rhb, hphi1, hphi2, d0, d1, d2, d3, d4, d5, d6, d7, pf0, pf1, pf2,
    clkp, ref_bar;
output cntd, pf;

wire cnt, rhb, hphi1, hphi2, d0, d1, d2, d3, d4, d5, d6, d7, pf0, pf1, pf2,
    clkp;
wire cntd, pf;

tia_d1 d1_cntd(.in(cnt), .s1(hphi1), .s2(hphi2), .out(cntd));

wire repeat_start;
assign repeat_start = (ref_bar & cnt) | rhb;
wire reflect_start;
assign reflect_start = ~(ref_bar | (~cnt));

// ==== PF0

wire pf0_bar;
assign pf0_bar = ~pf0;
wire pf0_bit5_so1, pf0_bit6_so1, pf0_bit7_so1;
wire pf0_bit4_so2, pf0_bit5_so2, pf0_bit6_so2, pf0_bit7_so2;
wire pf1_bit7_so2;
wire pf0_bit4_o, pf0_bit5_o, pf0_bit6_o, pf0_bit7_o;
wire pf0_bit;
assign pf0_bit = pf0_bit4_o | pf0_bit5_o | pf0_bit6_o | pf0_bit7_o;
// PF0 Bit 4 has no input delay bit on the s1 side, so we implement it directly.
wire pf0_bit4_l_val;
tia_l pf0_bit4_latch(.in(d4), .follow(pf0), .latch(pf0_bar),
    .out(pf0_bit4_l_val));
tia_d1 pf0_bit4_d1(.in(pf0_bit5_so2), .s1(hphi1), .s2(hphi2),
    .out(pf0_bit4_so2));
assign pf0_bit4_o = pf0_bit4_l_val & (repeat_start | pf0_bit4_so2);
tia_playfield_registers_cell pf0_bit5(.i(d5), .l1(pf0), .l2(pf0_bar),
    .si1(repeat_start), .si2(pf0_bit6_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf0_bit5_so1), .so2(pf0_bit5_so2), .o(pf0_bit5_o));
tia_playfield_registers_cell pf0_bit6(.i(d6), .l1(pf0), .l2(pf0_bar),
    .si1(pf0_bit5_so1), .si2(pf0_bit7_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf0_bit6_so1), .so2(pf0_bit6_so2), .o(pf0_bit6_o));
tia_playfield_registers_cell pf0_bit7(.i(d7), .l1(pf0), .l2(pf0_bar),
    .si1(pf0_bit6_so1), .si2(pf1_bit7_so1), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf0_bit7_so1), .so2(pf0_bit7_so2), .o(pf0_bit7_o));

// ==== PF1

wire pf1_bar;
assign pf1_bar = ~pf1;
wire pf1_bit0_so1, pf1_bit1_so1, pf1_bit2_so1, pf1_bit3_so1,
     pf1_bit4_so1, pf1_bit5_so1, pf1_bit6_so1, pf1_bit7_so1;
wire pf1_bit0_so2, pf1_bit1_so2, pf1_bit2_so2, pf1_bit3_so2,
     pf1_bit4_so2, pf1_bit5_so2, pf1_bit6_so2;
wire pf2_bit7_so2;
wire pf1_bit0_o, pf1_bit1_o, pf1_bit2_o, pf1_bit3_o,
     pf1_bit4_o, pf1_bit5_o, pf1_bit6_o, pf1_bit7_o;
wire pf1_bit;
assign pf1_bit = pf1_bit0_o | pf1_bit1_o | pf1_bit2_o | pf1_bit3_o |
                 pf1_bit4_o | pf1_bit5_o | pf1_bit6_o | pf1_bit7_o;
tia_playfield_registers_cell pf1_bit0(.i(d0), .l1(pf1), .l2(pf1_bar),
    .si1(pf2_bit0_so2), .si2(pf1_bit1_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf1_bit0_so1), .so2(pf1_bit0_so2), .o(pf1_bit0_o));
tia_playfield_registers_cell pf1_bit1(.i(d1), .l1(pf1), .l2(pf1_bar),
    .si1(pf1_bit0_so1), .si2(pf1_bit2_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf1_bit1_so1), .so2(pf1_bit1_so2), .o(pf1_bit1_o));
tia_playfield_registers_cell pf1_bit2(.i(d2), .l1(pf1), .l2(pf1_bar),
    .si1(pf1_bit1_so1), .si2(pf1_bit3_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf1_bit2_so1), .so2(pf1_bit2_so2), .o(pf1_bit2_o));
tia_playfield_registers_cell pf1_bit3(.i(d3), .l1(pf1), .l2(pf1_bar),
    .si1(pf1_bit2_so1), .si2(pf1_bit4_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf1_bit3_so1), .so2(pf1_bit3_so2), .o(pf1_bit3_o));
tia_playfield_registers_cell pf1_bit4(.i(d4), .l1(pf1), .l2(pf1_bar),
    .si1(pf1_bit3_so1), .si2(pf1_bit5_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf1_bit4_so1), .so2(pf1_bit4_so2), .o(pf1_bit4_o));
tia_playfield_registers_cell pf1_bit5(.i(d5), .l1(pf1), .l2(pf1_bar),
    .si1(pf1_bit4_so1), .si2(pf1_bit6_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf1_bit5_so1), .so2(pf1_bit5_so2), .o(pf1_bit5_o));
tia_playfield_registers_cell pf1_bit6(.i(d6), .l1(pf1), .l2(pf1_bar),
    .si1(pf1_bit5_so1), .si2(pf1_bit7_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf1_bit6_so1), .so2(pf1_bit6_so2), .o(pf1_bit6_o));
tia_playfield_registers_cell pf1_bit7(.i(d7), .l1(pf1), .l2(pf1_bar),
    .si1(pf1_bit6_so1), .si2(pf0_bit7_so1), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf1_bit7_so1), .so2(pf1_bit7_so2), .o(pf1_bit7_o));

// ==== PF2

wire pf2_bar;
assign pf2_bar = ~pf2;
wire pf2_bit0_so1, pf2_bit1_so1, pf2_bit2_so1, pf2_bit3_so1,
     pf2_bit4_so1, pf2_bit5_so1, pf2_bit6_so1, pf2_bit7_so1;
wire pf2_bit0_so2, pf2_bit1_so2, pf2_bit2_so2, pf2_bit3_so2,
     pf2_bit4_so2, pf2_bit5_so2, pf2_bit6_so2;
wire pf2_bit0_o, pf2_bit1_o, pf2_bit2_o, pf2_bit3_o,
     pf2_bit4_o, pf2_bit5_o, pf2_bit6_o, pf2_bit7_o;
wire pf2_bit;
assign pf2_bit = pf2_bit0_o | pf2_bit1_o | pf2_bit2_o | pf2_bit3_o |
                 pf2_bit4_o | pf2_bit5_o | pf2_bit6_o | pf2_bit7_o;
tia_playfield_registers_cell pf2_bit0(.i(d0), .l1(pf2), .l2(pf2_bar),
    .si1(pf1_bit0_so2), .si2(pf2_bit1_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf2_bit0_so1), .so2(pf2_bit0_so2), .o(pf2_bit0_o));
tia_playfield_registers_cell pf2_bit1(.i(d1), .l1(pf2), .l2(pf2_bar),
    .si1(pf2_bit0_so1), .si2(pf2_bit2_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf2_bit1_so1), .so2(pf2_bit1_so2), .o(pf2_bit1_o));
tia_playfield_registers_cell pf2_bit2(.i(d2), .l1(pf2), .l2(pf2_bar),
    .si1(pf2_bit1_so1), .si2(pf2_bit3_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf2_bit2_so1), .so2(pf2_bit2_so2), .o(pf2_bit2_o));
tia_playfield_registers_cell pf2_bit3(.i(d3), .l1(pf2), .l2(pf2_bar),
    .si1(pf2_bit2_so1), .si2(pf2_bit4_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf2_bit3_so1), .so2(pf2_bit3_so2), .o(pf2_bit3_o));
tia_playfield_registers_cell pf2_bit4(.i(d4), .l1(pf2), .l2(pf2_bar),
    .si1(pf2_bit3_so1), .si2(pf2_bit5_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf2_bit4_so1), .so2(pf2_bit4_so2), .o(pf2_bit4_o));
tia_playfield_registers_cell pf2_bit5(.i(d5), .l1(pf2), .l2(pf2_bar),
    .si1(pf2_bit4_so1), .si2(pf2_bit6_so2), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf2_bit5_so1), .so2(pf2_bit5_so2), .o(pf2_bit5_o));
tia_playfield_registers_cell pf2_bit6(.i(d6), .l1(pf2), .l2(pf2_bar),
    .si1(pf2_bit5_so1), .si2(reflect_start), .hphi1(hphi1), .hphi2(hphi2),
    .so1(pf2_bit6_so1), .so2(pf2_bit6_so2), .o(pf2_bit6_o));
// PF0 Bit 4 has no input delay bit on the s1 side, so we implement it directly.
wire pf2_bit7_l_val;
tia_l pf2_bit7_latch(.in(d7), .follow(pf2), .latch(pf2_bar),
    .out(pf2_bit7_l_val));
tia_d1 pf2_bit7_d1(.in(pf2_bit6_so1), .s1(hphi1), .s2(hphi2),
    .out(pf2_bit7_so1));
assign pf2_bit7_o = pf2_bit7_l_val & (reflect_start | pf2_bit7_so1);

wire pf_bit_d1_in;
assign pf_bit_d1_in = ~(pf0_bit | pf1_bit | pf2_bit);
wire pf_bit_f1_s;
tia_d1 pf_bit_d1(.in(pf_bit_d1_in), .s1(hphi1), .s2(hphi2), .out(pf_bit_f1_s));
wire pf_bit_f1_r;
assign pf_bit_f1_r = ~pf_bit_f1_s;
tia_f1 pf_bit_f1(.s(pf_bit_f1_s), .r(pf_bit_f1_r), .clock(clkp), .reset(0),
    .q(pf));

always @(pf_bit_d1_in, pf_bit_f1_s, pf) begin
  $display("pf_bit_d1_in: %d, pf_bit_f1_s: %d, pf: %d",
      pf_bit_d1_in, pf_bit_f1_s, pf);
end

/*
always @(posedge hphi2) begin
  #1
  $display("%d%d%d%d %d%d%d%d%d%d%d%d %d%d%d%d%d%d%d%d %d",
      repeat_start, pf0_bit5_so1, pf0_bit6_so1, pf0_bit7_so1,
      pf1_bit7_so2, pf1_bit6_so2, pf1_bit5_so2, pf1_bit4_so2,
      pf1_bit3_so2, pf1_bit2_so2, pf1_bit1_so2, pf1_bit0_so2,
      pf2_bit0_so1, pf2_bit1_so1, pf2_bit2_so1, pf2_bit3_so1,
      pf2_bit4_so1, pf2_bit5_so1, pf2_bit6_so1, pf2_bit7_so1,
      pf);
end
*/
endmodule  // tia_playfield_registers
