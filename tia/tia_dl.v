`ifndef TIA_TIA_DL_V
`define TIA_TIA_DL_V

// DL block defined on TIA schematics page 1, section C-1. A 1 clocked through
// s1 and s2 will remain latched until reset.
module tia_dl(
    input in,
    input s1,
    input s2,
    input r,
    output out);

wire in;
wire s1;
wire s2;
wire r;
reg latched_in;
reg latched_out;
wire out;

// TODO: behavioral description?

// If s2 is high then output is stored latch, otherwise it sees a zero.
assign #1 out = (~r) & latched_out;

initial begin
  latched_in = 0;
  latched_out = 0;
end

always @(s1 or in or out) begin
  if (s1) begin
    latched_in <= ~(in | out);
  end
end

always @(posedge s2) begin
  latched_out <= ~latched_in;
end

endmodule  // tia_d2

`endif  // TIA_TIA_DL_V
