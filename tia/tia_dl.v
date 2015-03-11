`ifndef TIA_TIA_DL_V
`define TIA_TIA_DL_V

// DL block defined on TIA schematics page 1, section C-1. A 1 clocked through
// s1 and s2 will remain latched until reset.
module tia_dl(in, s1, s2, r, out);

input in, s1, s2, r;
output out;

wire in;
wire s1;
wire s2;
wire r;
reg latched;
wire out;

// If s2 is high then output is stored latch, otherwise it sees a zero.
assign out = (r === 1) ? 0 : latched;

initial begin
  latched = 0;
end

always @(r) begin
  latched = 0;
end

always @(posedge s1, in) begin
  if ((s1 === 1) && (in === 1 || latched === 1)) latched = 1;
end

endmodule  // tia_d2

`endif  // TIA_TIA_DL_V
