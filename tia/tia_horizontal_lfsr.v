`include "tia_d1.v"
`include "tia_lfsr.v"

// Horizontal Linear Feedback Shift Register, used as a counter for various
// horizontal timings. Defined on TIA schematics page 1, section D-4 and D-3.
module tia_horizontal_lfsr(hphi1, hphi2, rsynl, out, shb, rsynd);

input hphi1, hphi2, rsynl;
output[5:0] out;
output shb, rsynd;

wire hphi1;
wire hphi2;
wire rsynl;
wire shb;
wire err;
wire wend;
wire d1_in;
wire d1_out;
wire rsynd;

tia_lfsr lfsr(.s1(hphi1), .s2(hphi2), .reset(shb), .out(out));

tia_d1 d1(.in(d1_in), .s1(hphi1), .s2(hphi2), .tap(rsynd), .out(d1_out));

assign err = &(out);
assign wend = (~out[5]) & out[4] & (~out[3]) & out[2] & (~out[1]) & (~out[0]);
assign d1_in = ~(wend | err | rsynl);
assign shb = ~d1_out;

endmodule  // tia_horizontal_lfsr
