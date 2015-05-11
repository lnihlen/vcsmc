`ifndef TIA_TIA_F1_V
`define TIA_TIA_F1_V

`include "sr.v"

// F1 block defined on TIA schematics page 1, section C-1.
module tia_f1(s, r, clock, reset, q, q_bar);

input s, r, clock, reset;
output q, q_bar;

wire s, r, clock, reset;
wire q, q_bar;

wire sr_a_s;
assign #1 sr_a_s = ~(s | clock);
wire sr_a_r;
assign #1 sr_a_r = ~(r | clock);
wire sr_a_q, sr_a_q_bar;
sr sr_a(.s(sr_a_s), .r(sr_a_r), .r2(reset), .q(sr_a_q), .q_bar(sr_a_q_bar));

wire sr_b_r;
assign #1 sr_b_r = clock & sr_a_q;
wire sr_b_s;
assign #1 sr_b_s = clock & sr_a_q_bar;
sr sr_b(.s(sr_b_s), .r(sr_b_r), .r2(reset), .q(q_bar), .q_bar(q));

endmodule  // tia_f1

`endif  // TIA_TIA_F1_V
