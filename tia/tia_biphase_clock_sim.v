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
  Z0    = 1,
  HPHI1 = 2,
  Z1    = 3,
  HPHI2 = 4,
  Z2    = 5;

tia_biphase_clock bpc(.clk(clock),
                      .rsyn(rsyn),
                      .hphi1(hphi1),
                      .hphi2(hphi2),
                      .rsynl(rsynl));

initial begin
  clock = 1'bz;
  rsyn = 1;
  state = RESET;
end

always #100 begin
  rsyn = 0;
  if (clock) begin
    clock = 1'bz;
  end else begin
    clock = 1;
  end
end

always @(posedge clock) begin
  case (state)
    RESET: begin
      if (rsynl === 0) state = Z0;
    end
    Z0: begin
      if (hphi1) state = HPHI1;
    end
    HPHI1: begin
      if (hphi1 === 1'bz) state = Z1;
    end
    Z1: begin
      if (hphi2) state = HPHI2;
    end
    HPHI2: begin
      if (hphi2 === 1'bz) state = Z2;
    end
    Z2: begin
      if (hphi1) begin
        $display("OK");
        $finish;
      end
    end
  endcase
end

always @(clock) begin
  #1
  case (state)
    RESET: begin
    end
    Z0: begin
      if (hphi1 != 1'bz || hphi2 != 1'bz || rsynl != 0) begin
        $display("ERROR in Z0");
        $finish;
      end
    end
    HPHI1: begin
      if (hphi1 != 1 || hphi2 != 1'bz || rsynl != 0) begin
        $display("ERROR in HPHI1");
        $finish;
      end
    end
    Z1: begin
      if (hphi1 != 1'bz || hphi2 != 1'bz || rsynl != 0) begin
        $display("ERROR in Z1");
        $finish;
      end
    end
    HPHI2: begin
      if (hphi1 != 1'bz || hphi2 != 1 || rsynl != 0) begin
        $display("ERROR in HPHI2");
        $finish;
      end
    end
    Z2: begin
      if (hphi1 != 1'bz || hphi2 != 1'bz || rsynl != 0) begin
        $display("ERROR in Z2");
        $finish;
      end
    end
  endcase
end

endmodule  // tia_biphase_clock_sim
