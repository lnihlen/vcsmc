`ifndef TIA_TIA_BIPHASE_CLOCK_V
`define TIA_TIA_BIPHASE_CLOCK_V

// Biphase clock splits input clock into two output clocks, used throughout TIA
// circuitry. Circuit block is defined on page 1 of TIA schematics, section D-5.
`include "sr.v"
`include "tia_f1.v"

module tia_biphase_clock(clk, r, phi1, phi2, bqb, rl);

input clk, r;

output phi1, phi2, bqb, rl;

wire clk, r;
wire phi1, phi2, rl;
wire a_q, a_q_bar, b_q, b_q_bar, b_q_bar_bar;

tia_f1 f1_a(.s(b_q),
            .r(b_q_bar),
            .clock(clk),
            .reset(r),
            .q(a_q),
            .q_bar(a_q_bar));

tia_f1 f1_b(.s(a_q_bar),
            .r(a_q),
            .clock(clk),
            .reset(r),
            .q(b_q),
            .q_bar(b_q_bar));

sr rsyn_sr(.s(b_q_bar_bar), .r(r), .r2(0), .q(rl));

assign phi1 = ~(a_q | b_q);
assign phi2 = ~(a_q_bar | b_q_bar);
assign b_q_bar_bar = ~b_q_bar;
// |bqb| is optional output, used only in missile position counter.
assign bqb = b_q_bar;

endmodule  // tia_biphase_clock

`endif  // TIA_TIA_BIPHASE_CLOCK_V
