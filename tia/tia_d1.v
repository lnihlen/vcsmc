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
reg out_store;
wire out;

assign out = (s2 === 1) ? out_store : 1;

initial begin
  tap = 1;
  out_store = 0;
end

always @(posedge s1, in) begin
  if (s1 === 1) begin
    if (in === 1) tap = 0; else tap = 1;
  end
end


always @(posedge s2) begin
  out_store = ~tap;
end

endmodule  // tia_d1

`endif  // TIA_TIA_D1_V
