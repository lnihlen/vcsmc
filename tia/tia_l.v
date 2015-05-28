`ifndef TIA_TIA_L
`define TIA_TIA_L

module tia_l(in, follow, latch, out, out_bar);
  input in, follow, latch;
  output out, out_bar;
  reg latch_reg;

  initial begin
    latch_reg = 0;
  end

  assign #1 out = follow ? in : latch_reg;
  assign out_bar = ~out;

  always @(follow, in) begin
    if (follow === 1) latch_reg = in;
  end
endmodule  // tia_l

`endif  // TIA_TIA_L
