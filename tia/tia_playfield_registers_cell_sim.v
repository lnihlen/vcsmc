`include "tia_biphase_clock.v"
`include "tia_playfield_registers_cell.v"

module tia_playfield_registers_cell_sim();

reg clock;
reg rsyn;
reg d0;
reg d1;
reg d2;
reg follow;
reg upper_si1;
reg lower_si2;
wire latch;
wire rsynl;
assign latch = ~follow;
wire hphi1, hphi2;
wire upper_so1, upper_so2;
wire middle_so1, middle_so2;
wire lower_so1, lower_so2;
wire upper_o;
wire middle_o;
wire lower_o;
wire out;
assign out = (upper_o | middle_o | lower_o);

integer cycle_count;

tia_biphase_clock bpc(.clk(clock),
                      .r(rsyn),
                      .phi1(hphi1),
                      .phi2(hphi2),
                      .rl(rsynl));

tia_playfield_registers_cell upper(.i(d0),
                                   .l1(follow),
                                   .l2(latch),
                                   .si1(upper_si1),
                                   .si2(middle_so2),
                                   .hphi1(hphi1),
                                   .hphi2(hphi2),
                                   .so1(upper_so1),
                                   .so2(upper_so2),
                                   .o(upper_o));

tia_playfield_registers_cell middle(.i(d1),
                                    .l1(follow),
                                    .l2(latch),
                                    .si1(upper_so1),
                                    .si2(lower_so2),
                                    .hphi1(hphi1),
                                    .hphi2(hphi2),
                                    .so1(middle_so1),
                                    .so2(middle_so2),
                                    .o(middle_o));

tia_playfield_registers_cell lower(.i(d2),
                                   .l1(follow),
                                   .l2(latch),
                                   .si1(middle_so1),
                                   .si2(lower_si2),
                                   .hphi1(hphi1),
                                   .hphi2(hphi2),
                                   .so1(lower_so1),
                                   .so2(lower_so2),
                                   .o(lower_o));

initial begin
  clock = 0;
  rsyn = 0;
  d0 = 0;
  d1 = 0;
  d2 = 0;
  follow = 0;
  cycle_count = 0;
  upper_si1 = 0;
  lower_si2 = 0;
end

always #100 begin
  clock = ~clock;
end

always @(posedge hphi1) begin
  case (cycle_count)
    0: begin
      follow = 1;
    end
    1: begin
      follow = 0;
      upper_si1 = 1;
    end
    2: begin
      upper_si1 = 0;
    end
    5: begin
      lower_si2 = 1;
    end
    6: begin
      lower_si2 = 0;
    end
    9: begin
      follow = 1;
      d0 = 1;
    end
    10: begin
      follow = 0;
      upper_si1 = 1;
    end
    11: begin
      upper_si1 = 0;
    end
    14: begin
      lower_si2 = 1;
    end
    15: begin
      lower_si2 = 0;
    end
  endcase
end

always @(posedge hphi2) begin
  if (cycle_count == 11 || cycle_count == 17) begin
    if (out != 1) begin
      $display("ERROR out: %d, cycle_count: %d", out, cycle_count);
    end
  end else begin
    if (out != 0) begin
      $display("ERROR out nonzero: %d, cycle_count: %d", out, cycle_count);
      $finish;
    end
  end

  if (cycle_count > 25) begin
    $display("OK");
    $finish;
  end
  cycle_count = cycle_count + 1;
end

endmodule  // tia_playfield_registers_cell_sim
