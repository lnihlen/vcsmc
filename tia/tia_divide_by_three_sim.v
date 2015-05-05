`include "tia_divide_by_three.v"

module tia_divide_by_three_sim();

reg clock;
reg resphi0;
wire phi_theta;

integer cycle_count;
integer last_positive_edge;

tia_divide_by_three d3(.clk(clock), .resphi0(resphi0), .phi_theta(phi_theta));

initial begin
  clock = 1;
  resphi0 = 0;
  cycle_count = 0;
  last_positive_edge = 0;
end

always #100 begin
  clock = ~clock;
  #1
  if (clock) cycle_count = cycle_count + 1;
end

always @(posedge phi_theta) begin
  // Wait for the first cycle to pass for the hardware to be in a correct state.
  if (cycle_count > 0 && last_positive_edge == 0) begin
    last_positive_edge = cycle_count;
  end else if (cycle_count > 300) begin
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
