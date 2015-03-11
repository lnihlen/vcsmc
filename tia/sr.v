`ifndef TIA_TIA_SR_V
`define TIA_TIA_SR_V

// SR latch, used throughout the TIA. Supports an optional additional reset
// input.
module sr(s, r, r2, q, q_bar);

input s, r, r2;
output q, q_bar;

wire s;
wire r;
wire r2;
reg q;
wire q_bar;

assign q_bar = ~q;

initial begin
  q = 0;
end

wire r_int;
assign r_int = (r === 1 || r2 === 1) ? 1 : 0;

always @(s, r_int) begin
  if (s === 0 && r_int === 1)
    q = 1;
  else if (s === 1 && r_int === 0)
    q = 0;
end

endmodule  // sr

`endif  // TIA_TIA_SR_V
