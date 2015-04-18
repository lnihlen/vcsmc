`ifndef TIA_TIA_MISSILE_POSITION_COUNTER_V
`define TIA_TIA_MISSILE_POSITION_COUNTER_V

`include "tia_biphase_clock.v"
`include "tia_d1.v"
`include "tia_f1.v"
`include "tia_lfsr.v"

module tia_missile_position_counter(
    // input
    motck,
    mec_bar,
    clkp,
    nz0_bar,
    nz1_bar,
    nz2_bar,
    missile_enable,
    missile_to_player_reset_bar,
    nz4_bar,
    nz5_bar,
    missile_reset,
    missile_to_player_reset,
    // output
    m
);

  input motck, mec_bar, clkp, nz0_bar, nz1_bar, nz2_bar, missile_enable,
      missile_to_player_reset_bar, nz4_bar, nz5_bar, missile_reset,
      missile_to_player_reset;
  output m;
  wire motck, mec_bar, clkp, nz0_bar, nz1_bar, nz2_bar, missile_enable,
      missile_to_player_reset_bar, nz4_bar, nz5_bar, missile_reset,
      missile_to_player_reset;
  wire m;

  wire missile_clock, mir, mphi1, mphi2, mr, bqb;
  assign missile_clock = (~mec_bar) | motck;
  assign mir = missile_to_player_reset | missile_reset;
  tia_biphase_clock bpc(.clk(missile_clock),
                        .r(mir),
                        .phi1(mphi1),
                        .phi2(mphi2),
                        .bqb(bqb),
                        .rl(mr));

  wire[5:0] out;
  wire lfsr_reset;
  tia_lfsr lfsr(.s1(mphi1), .s2(mphi2), .reset(lfsr_reset), .out(out));

  wire err, wend, rdin;
  assign err = &(out);
  assign wend = out[5] & (~out[4]) & out[3] & out[2] & (~out[1]) & out[0];
  assign rdin = ~(mr | wend | err);

  wire rdout;
  tia_d1 rd1(.in(rdin), .s1(mphi1), .s2(mphi2), .out(rdout));

  assign lfsr_reset = ~rdout;

  wire nz0, nz1, nz2, nz4, nz5;
  assign nz0 = ~nz0_bar;
  assign nz1 = ~nz1_bar;
  assign nz2 = ~nz2_bar;
  assign nz4 = ~nz4_bar;
  assign nz5 = ~nz5_bar;

  wire mns_bar;
  assign mns_bar = nz0_bar | nz2_bar;
  wire cntr_a, cntr_b, cntr_c;
  assign cntr_a = out[5] & out[4] & mns_bar & out[3] & (~out[2]) & (~out[1]) &
      nz0 & (~out[0]);
  assign cntr_b = out[5] & (~out[4]) & mns_bar & out[3] & out[2] & nz1 & out[1]
      & out[0];
  assign cntr_c = out[5] & out[4] & mns_bar & out[3] & nz2 & (~out[2]) &
      (~out[1]) & out[0];
  wire cd1in;
  assign cd1in = ~(wend | cntr_a | cntr_b | cntr_c);

  wire md1in, md1out;
  tia_d1 cd1(.in(cd1in), .s1(mphi1), .s2(mphi2), .out(md1in));
  tia_d1 md1(.in(md1in), .s1(mphi1), .s2(mphi2), .out(md1out));

  wire me;
  assign me = ~(missile_enable & missile_to_player_reset_bar);

  wire mf2in, mf2in_bar;
  assign mf2in = ~(~(~(~(bqb | nz4_bar) | mphi2 | nz5) | me | md1in) |
      ~(me | nz5_bar | md1out | nz4_bar));
  assign mf2in_bar = ~mf2in;

  tia_f1 mf1(.s(mf2in), .r(mf2in_bar), .clock(clkp), .reset(0), .q(m));

endmodule  // tia_missile_position_counter

`endif  // TIA_TIA_MISSILE_POSITION_COUNTER_V
