`include "tia_d1.v"

// D1R block defined on TIA schematics page 1, section D-1
module tia_d1r(in, s1, s2, r, out);

input in, s1, s2, r;
output out;

wire in;
wire s1;
wire s2;
wire r;
reg d;
wire out;
wire d1_out;

tia_d1 d1(.in(in), .s1(s1), .s2(s2), .out(d1_out));

assign out = (~r) & d1_out;

endmodule  // tia_d1r
