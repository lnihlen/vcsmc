`include "tia_divide_by_three.v"

module tia_divide_by_three_sim();

reg clock;
reg resphi0;
reg rsyn;
wire phi_theta, clkp, rsyn_gated;

integer cycle_count;
integer last_positive_edge;

tia_divide_by_three d3(.clk(clock), .resphi0(resphi0), .rsyn(rsyn),
    .phi_theta(phi_theta), .clkp(clkp), .rsyn_gated(rsyn_gated));

initial begin
  clock = 1;
  resphi0 = 0;
  rsyn = 0;
  cycle_count = 0;
  last_positive_edge = 0;
end

always #100 begin
  if (clock) begin
    clock = 1'bz;
  end else begin
    clock = 1;
    cycle_count = cycle_count + 1;
  end
end

always @(posedge phi_theta) begin
  // Wait for the first cycle to pass for the hardware to be in a correct state.
  if (cycle_count > 0 && last_positive_edge == 0) begin
    last_positive_edge = cycle_count;
  end else if (cycle_count > 30) begin
    $display("OK");
    $finish;
  end else if (last_positive_edge > 0) begin
    if (cycle_count - last_positive_edge != 3) begin
      $display("ERROR - cycle_count: %d, last_positive_edge: %d",
          cycle_count, last_positive_edge);
      $finish;
    end
    last_positive_edge = cycle_count;
  end
end

endmodule  // tia_divide_by_three_sim
