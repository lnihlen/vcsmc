`ifndef TIA_TIA_D2_V
`define TIA_TIA_D2_V

// D2 block defined on TIA schematics page 1, section D-1. |tap| is an optional
// output from the middle (inverted) state.
module tia_d2(in1, in2, s1, s2, tap, out);

input in1, in2, s1, s2;
output tap, out;

wire in1;
wire in2;
wire s1;
wire s2;
reg tap;
reg out;

initial begin
  tap = 1;
  out = 0;
end

always @(s1 or in1 or in2) begin
  if (s1) begin
    tap <= ~(in1 | in2);
  end
end

always @(posedge s2) begin
  out <= ~tap;
end

endmodule  // tia_d2

`endif  // TIA_TIA_D2_v
