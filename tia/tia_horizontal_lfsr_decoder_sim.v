`include "tia_biphase_clock.v"
`include "tia_horizontal_lfsr.v"
`include "tia_horizontal_lfsr_decoder.v"

module tia_horizontal_lfsr_decoder_sim();

reg clock;
reg rsyn;
wire hphi1;
wire hphi2;
wire rsynl;
wire[5:0] out;
wire shb;
wire rsynd;

wire rhs;
wire cnt;
wire rcb;
wire shs;
wire lrhb;
wire rhb;

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

tia_horizontal_lfsr_decoder decode(.in(out),
                                   .rhs(rhs),
                                   .cnt(cnt),
                                   .rcb(rcb),
                                   .shs(shs),
                                   .lrhb(lrhb),
                                   .rhb(rhb));

initial begin
  clock = 0;
  cycle_count = 0;
  rsyn = 0;

  $dumpfile("out/tia_horizontal_lfsr_decoder_sim.vcd");
  $dumpvars(0, tia_horizontal_lfsr_decoder_sim);
end

always #100 begin
  clock = ~clock;
end

always @(posedge hphi1) begin
  #1
  // Counter should start with 0 and repeat every 57 cycles.
  if (cycle_count == 4 || cycle_count == (4 + 57)) begin
    if (rhs != 0 || cnt != 0 || rcb != 0 ||
        shs != 1 || lrhb != 0 || rhb != 0) begin
      $display("SHS rhs: %d, cnt: %d, rcb: %d, shs: %d, lrhb: %d, rhb: %d",
          rhs, cnt, rcb, shs, lrhb, rhb);
      $finish;
    end
  end else if (cycle_count == 8 || cycle_count == (8 + 57)) begin
    if (rhs != 1 || cnt != 0 || rcb != 0 ||
        shs != 0 || lrhb != 0 || rhb != 0) begin
      $display("rhs timing error.");
      $finish;
    end
  end else if (cycle_count == 12 || cycle_count == (12 + 57)) begin
    if (rhs != 0 || cnt != 0 || rcb != 1 ||
        shs != 0 || lrhb != 0 || rhb != 0) begin
      $display("rcb timing error.");
      $finish;
    end
  end else if (cycle_count == 16 || cycle_count == (16 + 57)) begin
    if (rhs != 0 || cnt != 0 || rcb != 0 ||
        shs != 0 || lrhb != 0 || rhb != 1) begin
      $display("rhb timing error.");
      $finish;
    end
  end else if (cycle_count == 18 || cycle_count == (18 + 57)) begin
    if (rhs != 0 || cnt != 0 || rcb != 0 ||
        shs != 0 || lrhb != 1 || rhb != 0) begin
      $display("lrhb timing error.");
      $finish;
    end
  end else if (cycle_count == 36 || cycle_count == (36 + 57)) begin
    if (rhs != 0 || cnt != 1 || rcb != 0 ||
        shs != 0 || lrhb != 0 || rhb != 0) begin
      $display("lrhb timing error.");
      $finish;
    end
  end else if (cycle_count == 114) begin
    $display("OK");
    $finish;
  end else begin
    if (rhs != 0 || cnt != 0 || rcb != 0 ||
        shs != 0 || lrhb != 0 || rhb != 0) begin
      $display("nonzero value on cycle %d", cycle_count);
      $finish;
    end
  end
  cycle_count = cycle_count + 1;
end

endmodule  // tia_horizontal_lfsr_sim
