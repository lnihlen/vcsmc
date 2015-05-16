`include "tia_biphase_clock.v"

module tia_biphase_clock_sim();

reg clock;
reg rsyn;
wire hphi1;
wire hphi2;
wire rsynl;

reg [3:0] state;

parameter[3:0]
    RESET = 0,
    HPHI2 = 1,
    Z1    = 2,
    HPHI1 = 3,
    Z2    = 4;

tia_biphase_clock bpc(.clk(clock),
                      .r(rsyn),
                      .phi1(hphi1),
                      .phi2(hphi2),
                      .rl(rsynl));

initial begin
  clock = 0;
  rsyn = 1;
  state = RESET;
end

always #100 begin
  rsyn = 0;
  clock = ~clock;
end

always @(posedge clock) begin
  #5
  case (state)
    RESET: begin
      if (!rsynl) state = HPHI1;
    end
    HPHI1: begin
      if (!hphi1) state = Z1;
    end
    Z1: begin
      if (hphi2) state = HPHI2;
    end
    HPHI2: begin
      if (!hphi2) state = Z2;
    end
    Z2: begin
      if (hphi1) begin
        $display("OK");
        $finish;
      end
    end
  endcase
end

always @(negedge clock) begin
  if (hphi1 && hphi2) begin
    $display("ERROR - hphi1 and hphi2 1 at the same time.");
    $finish;
  end
  #5
  case (state)
    RESET: begin
    end
    HPHI2: begin
      if (hphi1 != 0 || hphi2 != 1 || rsynl != 0) begin
        $display("ERROR in HPHI2: %d %d %d", hphi1, hphi2, rsynl);
        $finish;
      end
    end
    Z1: begin
      if (hphi1 != 0 || hphi2 != 0 || rsynl != 0) begin
        $display("ERROR in Z1");
        $finish;
      end
    end
    HPHI1: begin
      if (hphi1 != 1 || hphi2 != 0 || rsynl != 0) begin
        $display("ERROR in HPHI1");
        $finish;
      end
    end
    Z2: begin
      if (hphi1 != 0 || hphi2 != 0 || rsynl != 0) begin
        $display("ERROR in Z2");
        $finish;
      end
    end
  endcase
end

endmodule  // tia_biphase_clock_sim
