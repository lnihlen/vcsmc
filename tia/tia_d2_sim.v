`include "tia_biphase_clock.v"
`include "tia_d2.v"

module tia_d2_sim();

reg d2_in1;
reg d2_in2;
wire s1;
wire s2;
reg clock;
wire d2_out;
reg rsyn;
wire rsynl;
wire tap;

integer cycle_count;

tia_biphase_clock bpc(.clk(clock),
                      .r(rsyn),
                      .phi1(s1),
                      .phi2(s2),
                      .rl(rsynl));

tia_d2 d2(.in1(d2_in1),
          .in2(d2_in2),
          .s1(s1),
          .s2(s2),
          .tap(tap),
          .out(d2_out));

initial begin
  d2_in1 = 0;
  d2_in2 = 0;
  clock = 0;
  rsyn = 0;
  cycle_count = 0;

  $dumpfile("out/tia_d2_sim.vcd");
  $dumpvars(0, tia_d2_sim);
end

always #100 begin
  clock = ~clock;
end

always @(posedge s1) begin
  case (cycle_count)
    2: begin
      d2_in1 = 1;
      d2_in2 = 0;
    end
    3: begin
      d2_in1 = 0;
      d2_in2 = 0;
    end
    4: begin
      d2_in1 = 1;
      d2_in2 = 0;
    end
    6: begin
      d2_in1 = 0;
      d2_in2 = 0;
    end

    8: begin
      d2_in1 = 0;
      d2_in2 = 1;
    end
    9: begin
      d2_in1 = 0;
      d2_in2 = 0;
    end
    10: begin
      d2_in1 = 0;
      d2_in2 = 1;
    end
    12: begin
      d2_in1 = 0;
      d2_in2 = 0;
    end

    14: begin
      d2_in1 = 1;
      d2_in2 = 1;
    end
  endcase
end

always @(posedge s2) begin
  #1
  case (cycle_count)
    2: begin
      if (d2_out != 1 || tap != 0) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    3: begin
      if (d2_out != 0 || tap != 1) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    4: begin
      if (d2_out != 1 || tap != 0) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    5: begin
      if (d2_out != 1 || tap != 0) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    6: begin
      if (d2_out != 0 || tap != 1) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    7: begin
      if (d2_out != 0 || tap != 1) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end

    8: begin
      if (d2_out != 1 || tap != 0) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    9: begin
      if (d2_out != 0 || tap != 1) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    10: begin
      if (d2_out != 1 || tap != 0) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    11: begin
      if (d2_out != 1 || tap != 0) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    12: begin
      if (d2_out != 0 || tap != 1) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    13: begin
      if (d2_out != 0 || tap != 1) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    14: begin
      if (d2_out != 1 || tap != 0) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    15: begin
      if (d2_out != 1 || tap != 0) begin
        $display("error cycle: %d, d2_in1: %d, d2_in2: %d, d2_out: %d, tap: %d",
            cycle_count, d2_in1, d2_in2, d2_out, tap);
        $finish;
      end
    end
    16: begin
      $display("OK");
      $finish;
    end
  endcase
  cycle_count = cycle_count + 1;
end

endmodule  // tia_d1_sim
