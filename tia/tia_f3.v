`ifndef TIA_TIA_F3_V
`define TIA_TIA_F3_V

`include "tia_f1.v"

module tia_f3(s1, s2, r1, r2, clock, reset, q, q_bar);

input s1, s2, r1, r2, clock, reset;
output q, q_bar;

wire s1, s2, r1, r2, clock, reset;
wire q, q_bar;

wire s, r;
assign s = s1 | s2;
assign r = r1 | r2;
tia_f1 f1(.s(s), .r(r), .clock(clock), .reset(reset), .q(q), .q_bar(q_bar));

endmodule  // tia_f3

`endif  // TIA_TIA_F3_V
