`ifndef TIA_TIA_D1_V
`define TIA_TIA_D1_V

// D1 block defined on TIA schematics page 1, section D-1. |tap| is an optional
// output from the middle (inverted) state.
module tia_d1(in, s1, s2, tap, out);

input in, s1, s2;
output tap, out;

wire in;
wire s1;
wire s2;
reg tap;
reg out;

initial begin
  tap = 1;
  out = 0;
end

always @(s1 or in) begin
  if (s1) tap <= ~in;
end

always @(posedge s2) begin
  out <= ~tap;
end

endmodule  // tia_d1

`endif  // TIA_TIA_D1_V
