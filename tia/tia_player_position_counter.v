`include "tia_biphase_clock.v"
`include "tia_d1.v"
`include "tia_dl.v"
`include "tia_lfsr.v"

module tia_player_position_counter(
  // input
  motck,
  p0ec_bar,
  p0re,
  nz0,
  nz1,
  nz2,
  // output
  start_bar,
  fstob,
  pck,
  pphi1,
  pphi2,
  count_bar);

input motck, p0ec_bar, p0re, nz0, nz1, nz2;
output start_bar, fstob, pck, pphi1, pphi2, count_bar;
wire motck, p0ec_bar, p0re, nz0, nz1, nz2;
wire start_bar, fstob, pck, pphi1, pphi2, count_bar;

assign pck = (~p0ec_bar) | motck;

wire r;
tia_biphase_clock bpc(.clk(pck),
                      .rsyn(p0re),
                      .hphi1(pphi1),
                      .hphi2(pphi2),
                      .rsynl(r));
wire[5:0] out;
wire lfsr_reset;
tia_lfsr lfsr(.s1(pphi1), .s2(pphi2), .reset(lfsr_reset), .out(out));

wire err, wend, rdin;
assign err = &(out);
assign wend = out[5] & (~out[4]) & out[3] & out[2] & (~out[1]) & out[0];
assign rdin = ~(r | wend | err);

wire rdout;
tia_d1 rd1(.in(rdin), .s1(pphi1), .s2(pphi2), .out(rdout));

assign lfsr_reset = ~rdout;

wire pns;
assign pns = ~(nz0 | nz2);
wire pns_bar;
assign pns_bar = ~pns;
wire cntr_a, cntr_b, cntr_c;
assign cntr_a = out[5] & out[4] & pns_bar & out[3] & (~out[2]) & (~out[1]) &
    nz0 & (~out[0]);
assign cntr_b = out[5] & (~out[4]) & pns_bar & out[3] & out[2] & nz1 &
    out[1] & out[0];
assign cntr_c = out[5] & out[4] & pns_bar & out[3] & nz2 & (~out[2]) &
    (~out[1]) & out[0];
wire fdlin;
assign fdlin = (cntr_a | cntr_b | cntr_c);
tia_dl fsdl(.in(fdlin), .s1(pphi1), .s2(pphi2), .r(lfsr_reset), .out(fstob));

wire sdin;
assign sdin = ~(wend | fdlin);
tia_d1 sd1(.in(sdin), .s1(pphi1), .s2(pphi2), .out(start_bar));

assign count_bar = ~(pphi2 | (pphi1 & nz1) | pns_bar);

endmodule  // tia_player_position_counter
