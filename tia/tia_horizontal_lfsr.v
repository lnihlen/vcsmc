`include "tia_d1r.v"

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
reg[5:0] in;
reg[5:0] out;

tia_d1 d1(.in(d1_in), .s1(hphi1), .s2(hphi2), .tap(rsynd), .out(d1_out));

assign err = &(out);
assign wend = (~out[5]) & out[4] & (~out[3]) & out[2] & (~out[1]) & (~out[0]);
assign d1_in = ~(wend | err | rsynl);
assign shb = d1_out ? 1'bz : 1;

initial begin
  in[5:0] = 6'b000000;
  out[5:0] = 6'b000000;
end

always @(shb) begin
  if (shb)
    assign out[5:0] = 6'b000000;
  else
    deassign out;
end

always @(posedge hphi1) begin
  in[4:0] = out[5:1];
  in[5] = out[1] ^ (~out[0]);
end

always @(posedge hphi2) begin
  out[5:0] = in[5:0];
end

endmodule  // tia_horizontal_lfsr
