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
wire out;
reg out_store;

// If s2 is high then output is stored latch, otherwise it sees a zero.
assign out = (s2 === 1) ? out_store : 1;

initial begin
  tap = 1;
  out_store = 0;
end

always @(posedge s1, in1, in2) begin
  if (s1 === 1) begin
    if (in1 === 1 || in2 === 1) tap = 0; else tap = 1;
  end
end

always @(posedge s2) begin
  out_store = ~tap;
end

endmodule  // tia_d2

`endif  // TIA_TIA_D2_v
