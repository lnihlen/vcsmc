`include "tia_d1r.v"

// Horizontal Linear Feedback Shift Register, used as a counter for various
// horizontal timings. Defined on TIA schematics page 1, section D-4 and D-3.
module tia_horizontal_lfsr(hphi1, hphi2, rsynl, a, b, c, d, e, f, shb, rsynd);

input hphi1, hphi2, rsynl;
output a, b, c, d, e, f, shb, rsynd;

wire hphi1;
wire hphi2;
wire rsynl;
wire a;
wire b;
wire c;
wire d;
wire e;
wire f;
wire shb;
wire g;
wire err;
wire wend;
wire d1_in;
wire d1_out;
wire rsynd;

tia_d1r d1r_a(.in(g), .s1(hphi1), .s2(hphi2), .r(shb), .out(a));
tia_d1r d1r_b(.in(a), .s1(hphi1), .s2(hphi2), .r(shb), .out(b));
tia_d1r d1r_c(.in(b), .s1(hphi1), .s2(hphi2), .r(shb), .out(c));
tia_d1r d1r_d(.in(c), .s1(hphi1), .s2(hphi2), .r(shb), .out(d));
tia_d1r d1r_e(.in(d), .s1(hphi1), .s2(hphi2), .r(shb), .out(e));
tia_d1r d1r_f(.in(e), .s1(hphi1), .s2(hphi2), .r(shb), .out(f));
tia_d1 d1(.in(d1_in), .s1(hphi1), .s2(hphi2), .tap(rsynd), .out(d1_out));

assign g = e ^ (~f);
assign err = (~a) & (~b) & (~c) & (~d) & (~e) & (~f);
assign wend = a & (~b) & c & (~d) & e & f;
assign d1_in = ~(wend | err | rsynl);
assign shb = d1_out ? 1'bz : 1;

always @(hphi1, hphi2, rsynl, a, b, c, d, e, f, g, shb, err, wend, d1_in, d1_out, rsynd) begin
  $display("hphi1: %d, hphi2: %d, rsynl: %d, a: %d, b: %d, c: %d, d: %d, e: %d, f: %d, g: %d, shb: %d, err: %d, wend: %d, d1_in: %d, d1_out: %d, rsynd: %d",
      hphi1, hphi2, rsynl, a, b, c, d, e, f, g, shb, err, wend, d1_in, d1_out, rsynd);
end

endmodule  // tia_horizontal_lfsr
