`include "tia_biphase_clock.v"
`include "tia_d1.v"

module tia_d1_sim();

reg d1_in;
wire s1;
wire s2;
reg clock;
wire d1_out;
reg rsyn;
wire rsynl;
wire tap;

integer cycle_count;

tia_biphase_clock bpc(.clk(clock),
                      .rsyn(rsyn),
                      .hphi1(s1),
                      .hphi2(s2),
                      .rsynl(rsynl));

tia_d1 d1(.in(d1_in), .s1(s1), .s2(s2), .tap(tap), .out(d1_out));

initial begin
  d1_in = 0;
  clock = 1'bz;
  rsyn = 0;
  cycle_count = 0;
end

always #100 begin
  if (clock) begin
    clock = 1'bz;
  end else begin
    clock = 1;
  end
end

always @(posedge s1) begin
  case (cycle_count)
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
  endcase
end

always @(posedge s2) begin
  #5
  case (cycle_count)
    2: begin
      if (d1_out != 1 || tap != 0) begin
        $display("ERROR cycle 1, d1_in: %d, d1_out: %d", d1_in, d1_out);
        $finish;
      end
    end
    3: begin
      if (d1_out != 0 || tap != 1) begin
        $display("ERROR cycle 2");
        $finish;
      end
    end
    4: begin
      if (d1_out != 1 || tap != 0) begin
        $display("ERROR cycle 3");
        $finish;
      end
    end
    5: begin
      if (d1_out != 1 || tap != 0) begin
        $display("ERROR cycle 4");
        $finish;
      end
    end
    6: begin
      if (d1_out != 0 || tap != 1) begin
        $display("ERROR cycle 5");
        $finish;
      end
    end
    7: begin
      if (d1_out != 0 || tap != 1) begin
        $display("ERROR cycle 6");
        $finish;
      end
    end
    8: begin
      $display("OK");
      $finish;
    end
  endcase
  cycle_count = cycle_count + 1;
end

endmodule  // tia_d1_sim
