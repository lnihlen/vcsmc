`ifndef TIA_TIA_BIPHASE_CLOCK_V
`define TIA_TIA_BIPHASE_CLOCK_V

// Biphase clock splits input clock into two output clocks, used throughout TIA
// circuitry. Circuit block is defined on page 1 of TIA schematics, section D-5.
`include "sr.v"
`include "tia_f1.v"

module tia_biphase_clock(clk, r, phi1, phi2, bqb, rl);

input clk, r;
output phi1, phi2, bqb, rl;

wire clk, r;
wire phi1, phi2, rl;

parameter[2:0]
  RESET = 0,
  PHI1_ON = 1,
  PHI1_OFF = 2,
  PHI2_ON = 3,
  PHI2_OFF = 4;

reg[2:0] state;

initial begin
  state <= RESET;
end

always @(posedge r) begin
  state <= RESET;
end

always @(posedge clk) begin
  if (state == RESET && r === 0) state <= PHI1_ON;
  else if (state == PHI1_ON) state <= PHI1_OFF;
  else if (state == PHI1_OFF) state <= PHI2_ON;
  else if (state == PHI2_ON) state <= PHI2_OFF;
  else if (state == PHI2_OFF) state <= PHI1_ON;
end

assign phi1 = state == PHI1_ON ? 1 : 0;
assign phi2 = state == PHI2_ON ? 1 : 0;
assign bqb = (state == PHI1_ON || state == PHI1_OFF) ? 1: 0;

wire bqb_n = ~bqb;
sr rlsr(.s(bqb_n), .r(r), .r2(0), .q(rl));

endmodule  // tia_biphase_clock

`endif  // TIA_TIA_BIPHASE_CLOCK_V
