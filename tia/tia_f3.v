`ifndef TIA_TIA_F3_V
`define TIA_TIA_F3_V

`include "sr.v"

module tia_f3(s1, s2, r1, r2, clock, reset, q, q_bar);

input s1, s2, r1, r2, clock, reset;
output q, q_bar;

wire s1, s2, r1, r2, clock, reset;
wire q, q_bar;

wire sr_a_s;
assign #1 sr_a_s = ~(s1 | s2 | clock);
wire sr_a_r;
assign #1 sr_a_r = ~(r1 | r2 | clock);
wire sr_a_q, sr_a_q_bar;
sr sr_a(.s(sr_a_s), .r(sr_a_r), .r2(reset), .q(sr_a_q), .q_bar(sr_a_q_bar));

wire sr_b_r;
assign #1 sr_b_r = clock & sr_a_q;
wire sr_b_s;
assign #1 sr_b_s = clock & sr_a_q_bar;
sr sr_b(.s(sr_b_s), .r(sr_b_r), .r2(reset), .q(q_bar), .q_bar(q));

endmodule  // tia_f3

`endif  // TIA_TIA_F3_V
