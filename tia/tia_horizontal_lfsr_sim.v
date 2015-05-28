`include "tia_biphase_clock.v"
`include "tia_horizontal_lfsr.v"

module tia_horizontal_lfsr_sim();

reg clock;
reg rsyn;
wire hphi1;
wire hphi2;
wire rsynl;
wire[5:0] out;
wire shb;
wire rsynd;

integer cycle_count;

tia_biphase_clock bpc(.clk(clock),
                      .r(rsyn),
                      .phi1(hphi1),
                      .phi2(hphi2),
                      .rl(rsynl));

tia_horizontal_lfsr lfsr(.hphi1(hphi1),
                         .hphi2(hphi2),
                         .rsynl(rsynl),
                         .out(out),
                         .shb(shb),
                         .rsynd(rsynd));

initial begin
  clock = 0;
  cycle_count = 0;
  rsyn = 0;

  $dumpfile("out/tia_horizontal_lfsr_sim.vcd");
  $dumpvars(0, tia_horizontal_lfsr_sim);
end

always #100 begin
  clock = ~clock;
end

always @(posedge hphi1) begin
  // Counter should start with 0 and repeat every 57 cycles.
  if (cycle_count == 0) begin
    if (out != 0) begin
      $display("ERROR: didn't start with 0, out: %b", out);
      $finish;
    end
  end else if (cycle_count == 57) begin
    if (out != 0) begin
      $display("ERROR, didn't cycle to 0 at 57, out: %b", out);
      $finish;
    end
  end else if (cycle_count == 114) begin
    if (out != 0) begin
      $display("ERROR, didn't cycle to 0 at 114, out: %b", out);
      $finish;
    end else begin
      $display("OK");
      $finish;
    end
  end else begin
    if (!out) begin
      $display("ERROR, zero output out-of-cycle %d.", cycle_count);
      $finish;
    end
  end
  cycle_count = cycle_count + 1;
end

endmodule  // tia_horizontal_lfsr_sim
