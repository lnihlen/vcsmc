`ifndef TIA_TIA_DIVIDE_BY_THREE_V
`define TIA_TIA_DIVIDE_BY_THREE_V
`include "sr.v"

module tia_divide_by_three(clk, resphi0, phi_theta);

input clk, resphi0;
output phi_theta;

wire clk, resphi0;
wire phi_theta;

wire q2;
wire q2_bar;
wire m1_sr_s, m1_sr_r;
wire m1, m1_sr_q;
assign m1_sr_s = ~(clk | q2);
assign m1_sr_r = ~(clk | q2_bar);
sr m1_sr(.s(m1_sr_s), .r(m1_sr_r), .r2(0), .q(m1_sr_q), .q_bar(m1));

assign phi_theta = ~(q2 | m1_sr_q);

wire q1, q1_bar;
wire q1_sr_s;
assign q1_sr_s = clk & m1_sr_q;
wire q1_sr_r;
assign q1_sr_r = clk & m1;
sr q1_sr(.s(q1_sr_s), .r(q1_sr_r), .r2(resphi0), .q(q1), .q_bar(q1_bar));

wire m2_mid;
assign m2_mid = ~(q1_bar | q2);
wire m2_sr_s;
assign m2_sr_s = ~(clk | ~(m2_mid));
wire m2_sr_r;
assign m2_sr_r = ~(clk | m2_mid);
wire m2, m2_sr_q;
sr m2_sr(.s(m2_sr_s), .r(m2_sr_r), .r2(0), .q(m2_sr_q), .q_bar(m2));

wire q2_sr_s;
assign q2_sr_s = m2_sr_q & clk;
wire q2_sr_r;
assign q2_sr_r = m2 & clk;
sr q2_sr(.s(q2_sr_s), .r(q2_sr_r), .r2(resphi0), .q(q2), .q_bar(q2_bar));

endmodule  // tia_divide_by_three

`endif  // TIA_TIA_DIVIDE_BY_THREE_V
