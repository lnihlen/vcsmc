// Biphase clock splits input clock into two output clocks, used throughout TIA
// circuitry. Circuit block is defined on page 1 of TIA schematics, section D-5.
`include "sr.v"
`include "tia_f1.v"

module tia_biphase_clock(clk, rsyn, hphi1, hphi2, rsynl);

input clk;
input rsyn;

output hphi1;
output hphi2;
output rsynl;

wire clk;
wire rsyn;
wire hphi1;
wire hphi2;
wire rsynl;

wire a_q;
wire a_q_bar;
wire b_q;
wire b_q_bar;
wire b_q_bar_bar;

tia_f1 f1_a(.s(b_q),
            .r(b_q_bar),
            .clock(clk),
            .reset(rsyn),
            .q(a_q),
            .q_bar(a_q_bar));

tia_f1 f1_b(.s(a_q_bar),
            .r(a_q),
            .clock(clk),
            .reset(rsyn),
            .q(b_q),
            .q_bar(b_q_bar));

sr rsyn_sr(.s(b_q_bar_bar), .r(rsyn), .q(rsynl));

assign hphi1 = (a_q === 1 || b_q === 1) ? 1'bz : 1;
assign hphi2 = (a_q_bar === 1 || b_q_bar === 1) ? 1'bz : 1;
assign b_q_bar_bar = ~b_q_bar;

endmodule  // tia_biphase_clock
