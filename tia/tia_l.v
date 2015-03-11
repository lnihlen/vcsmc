`ifndef TIA_TIA_L
`define TIA_TIA_L

module tia_l(in, follow, latch, out, out_bar);

input in, follow, latch;
output out, out_bar;
reg ltch;

initial begin
  ltch = 0;
end

assign out = (follow === 1) ? in : (latch === 1) ? ~ltch : 0;
assign out_bar = ~out;

always @(follow, in) begin
  if (follow === 1) ltch = ~in;
end

endmodule  // tia_l

`endif  // TIA_TIA_L
