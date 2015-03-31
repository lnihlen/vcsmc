`ifndef TIA_TIA_PLAYFIELD_REGISTERS_CELL_V_
`define TIA_TIA_PLAYFIELD_REGISTERS_CELL_V_

`include "tia_l.v"

// Actual playfield cells are in a wired OR configuration but we just
// conventionally OR them together, and drop the inverter at the top.
module tia_playfield_registers_cell(i, l1, l2, si1, si2, hphi1, hphi2,
    so1, so2, o);
  input i, l1, l2, si1, si2, hphi1, hphi2;
  output so1, so2, o;
  wire i, l1, l2, si1, si2, hphi1, hphi2;
  wire o;
  wire l_val;
  tia_l input_latch(.in(i), .follow(l1), .latch(l2), .out(l_val));
  reg si1_store, si2_store, so1, so2;
  initial begin
    si1_store = 0;
    si2_store = 0;
    so1 = 0;
    so2 = 0;
  end
  always @(posedge hphi1, si1, si2) begin
    if (hphi1) begin
      si1_store = si1;
      si2_store = si2;
    end
  end
  always @(posedge hphi2, si1_store, si2_store) begin
    if (hphi2) begin
      so1 = si1_store;
      so2 = si2_store;
    end
  end
  assign o = l_val & (so1 | so2);
endmodule  // tia_playfield_registers_cell

`endif  // TIA_TIA_PLAYFIELD_REGISTERS_CELL_V_
