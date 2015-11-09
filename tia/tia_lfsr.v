`ifndef TIA_TIA_LFSR_V
`define TIA_TIA_LFSR_V

// Linear Feedback Shift Register, used as a counter for various horizontal
// timings.
module tia_lfsr(s1, s2, reset, out);

input s1, s2, reset;
output[5:0] out;

wire s1, s2, reset;
reg[5:0] in;
reg[5:0] out;

initial begin
  in[5:0] = 6'b000000;
  out[5:0] = 6'b000000;
end

always @(posedge s1) begin
  in[4:0] <= out[5:1];
  in[5] <= out[1] ^ (~out[0]);
end

always @(posedge s2) begin
  out[5:0] <= in[5:0];
end

always @(posedge reset) begin
  in[5:0] <= 6'b000000;
  out[5:0] <= 6'b000000;
end

endmodule  // tia_horizontal_lfsr

`endif  // TIA_TIA_LFSR_V
