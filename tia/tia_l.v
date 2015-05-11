`ifndef TIA_TIA_L
`define TIA_TIA_L

module tia_l(in, follow, latch, out, out_bar);
  input in, follow, latch;
  output out, out_bar;
  reg ltch;

  initial begin
    ltch = 0;
  end

  assign #1 out = follow ? in : (latch ? ~ltch : 0);
  assign #1 out_bar = ~out;

  always @(follow, in) begin
    #1
    if (follow) ltch = ~in;
  end
endmodule  // tia_l

`endif  // TIA_TIA_L
