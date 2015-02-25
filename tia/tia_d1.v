// D1 block defined on TIA schematics page 1, section D-1. |tap| is an optional
// output from the middle (inverted) state.
module tia_d1(in, s1, s2, tap, out);

input in, s1, s2;
output tap, out;

wire in;
wire s1;
wire s2;
reg tap;
wire out;

// If s2 is high then output is stored latch, otherwise it sees a zero.
assign out = s2 ? ~tap : 1;

initial begin
  tap = 0;
end

always @(negedge s1) begin
  tap = ~in;
end

endmodule  // tia_d1
