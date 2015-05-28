`include "tia_biphase_clock.v"
`include "tia_d1r.v"

module tia_d1r_sim();

reg d1_in;
wire s1;
wire s2;
reg clock;
wire d1_out;
reg rsyn;
wire rsynl;
reg reset;

integer cycle_count;

tia_biphase_clock bpc(.clk(clock),
                      .r(rsyn),
                      .phi1(s1),
                      .phi2(s2),
                      .rl(rsynl));

tia_d1r d1r(.in(d1_in), .s1(s1), .s2(s2), .r(reset), .out(d1_out));

initial begin
  d1_in = 0;
  clock = 0;
  rsyn = 0;
  cycle_count = 0;
  reset = 0;

  $dumpfile("out/tia_d1r_sim.vcd");
  $dumpvars(0, tia_d1r_sim);
end

always #100 begin
  clock = ~clock;
end

always @(posedge s1) begin
  case (cycle_count)
    1: begin
      reset = 0;
    end
    2: begin
      d1_in = 1;
    end
    3: begin
      d1_in = 0;
    end
    4: begin
      d1_in = 1;
    end
    6: begin
      d1_in = 0;
    end
    8: begin
      reset = 1;
      d1_in = 1;
    end
    9: begin
      d1_in = 0;
    end
    10: begin
      d1_in = 1;
    end
    12: begin
      d1_in = 0;
    end
  endcase
end

always @(posedge s2) begin
  #10
  case (cycle_count)
    2: begin
      if (d1_out != 1) begin
        $display("ERROR cycle 2, d1_out: %d", d1_out);
        $finish;
      end
    end
    3: begin
      if (d1_out != 0) begin
        $display("ERROR cycle 3, d1_out: %d", d1_out);
        $finish;
      end
    end
    4: begin
      if (d1_out != 1) begin
        $display("ERROR cycle 4");
        $finish;
      end
    end
    5: begin
      if (d1_out != 1) begin
        $display("ERROR cycle 5");
        $finish;
      end
    end
    6: begin
      if (d1_out != 0) begin
        $display("ERROR cycle 6");
        $finish;
      end
    end
    7: begin
      if (d1_out != 0) begin
        $display("ERROR cycle 7");
        $finish;
      end
    end
  endcase
  if (cycle_count > 15) begin
    $display("OK");
    $finish;
  end else if (cycle_count > 7) begin
    if (d1_out != 0) begin
      $display("ERROR r = 1 but d1_out != 0");
      $finish;
    end
  end
  cycle_count = cycle_count + 1;
end

endmodule  // tia_d1r_sim
