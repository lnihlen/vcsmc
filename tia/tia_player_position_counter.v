`ifndef TIA_TIA_PLAYER_POSITION_COUNTER_V
`define TIA_TIA_PLAYER_POSITION_COUNTER_V

`include "tia_biphase_clock.v"
`include "tia_d1.v"
`include "tia_dl.v"
`include "tia_lfsr.v"

module tia_player_position_counter(
    // input
    motck,
    pec_bar,
    pre,
    nz0_bar,
    nz1_bar,
    nz2_bar,
    // output
    start_bar,
    fstob,
    pck,
    pphi1,
    pphi2,
    count_bar);

  input motck, pec_bar, pre, nz0_bar, nz1_bar, nz2_bar;
  output start_bar, fstob, pck, pphi1, pphi2, count_bar;
  wire motck, pec_bar, pre, nz0_bar, nz1_bar, nz2_bar;
  wire start_bar, fstob, pck, pphi1, pphi2, count_bar;

  assign pck = (~pec_bar) | motck;

  wire r;
  tia_biphase_clock bpc(.clk(pck),
                        .r(pre),
                        .phi1(pphi1),
                        .phi2(pphi2),
                        .rl(r));
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

  wire pns_bar;
  assign pns_bar = nz0_bar | nz2_bar;
  wire nz0, nz1, nz2;
  assign nz0 = ~nz0_bar;
  assign nz1 = ~nz1_bar;
  assign nz2 = ~nz2_bar;
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

  assign count_bar = ~(pphi2 | (pphi1 & nz1_bar) | pns_bar);

endmodule  // tia_player_position_counter

`endif  // TIA_TIA_PLAYER_POSITION_COUNTER_V
