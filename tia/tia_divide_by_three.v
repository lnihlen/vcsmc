`include "sr.v"

module tia_divide_by_three(clk, resphi0, rsyn, phi_theta, clkp, rsyn_gated);

input clk, resphi0, rsyn;
output phi_theta, clkp, rsyn_gated;

wire clk, resphi0, rsyn;
wire phi_theta, clkp, rsyn_gated;

// Build a clock internally that is always either 0 or 1, so that the continuous
// assignments are always either 0 or 1 too.
wire log_clk;
assign log_clk = (clk === 1) ? 1 : 0;

wire q2;
wire q2_bar;
wire m1_sr_s, m1_sr_r;
wire m1, m1_sr_q;
assign m1_sr_s = ~(log_clk | q2);
assign m1_sr_r = ~(log_clk | q2_bar);
sr m1_sr(.s(m1_sr_s), .r(m1_sr_r), .r2(0), .q(m1_sr_q), .q_bar(m1));

wire pp_in;
assign pp_in = ~(q2 | m1_sr_q);
assign phi_theta = (pp_in === 1) ? 1 : 0;

wire q1, q1_bar;
wire q1_sr_s;
assign q1_sr_s = log_clk & m1_sr_q;
wire q1_sr_r;
assign q1_sr_r = log_clk & m1;
sr q1_sr(.s(q1_sr_s), .r(q1_sr_r), .r2(resphi0), .q(q1), .q_bar(q1_bar));

wire m2_mid;
assign m2_mid = ~(q1_bar | q2);
wire m2_sr_s;
assign m2_sr_s = ~(log_clk | ~(m2_mid));
wire m2_sr_r;
assign m2_sr_r = ~(log_clk | m2_mid);
wire m2, m2_sr_q;
sr m2_sr(.s(m2_sr_s), .r(m2_sr_r), .r2(0), .q(m2_sr_q), .q_bar(m2));

assign clkp = (clk === 1) ? 1'bz : 1;

wire q2_sr_s;
assign q2_sr_s = m2_sr_q & log_clk;
wire q2_sr_r;
assign q2_sr_r = m2 & log_clk;
sr q2_sr(.s(q2_sr_s), .r(q2_sr_r), .r2(resphi0), .q(q2), .q_bar(q2_bar));

assign rsyn_gated = (phi_theta === 1) ? 0 : rsyn;

endmodule  // tia_divide_by_three
