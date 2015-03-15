`include "tia_biphase_clock.v"
`include "tia_horizontal_lfsr.v"
`include "tia_horizontal_lfsr_decoder.v"
`include "tia_playfield_registers.v"

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
                      .rsyn(rsyn),
                      .hphi1(hphi1),
                      .hphi2(hphi2),
                      .rsynl(rsynl));

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

reg d0, d1, d2, d3, d4, d5, d6, d7;
reg pf0, pf1, pf2;
wire clkp;
assign clkp = (clock === 1) ? 1'bz : 1;
reg ref_bar;
wire cntd, pf;

tia_playfield_registers pf_reg(.cnt(cnt),
                               .rhb(rhb),
                               .hphi1(hphi1),
                               .hphi2(hphi2),
                               .d0(d0),
                               .d1(d1),
                               .d2(d2),
                               .d3(d3),
                               .d4(d4),
                               .d5(d5),
                               .d6(d6),
                               .d7(d7),
                               .pf0(pf0),
                               .pf1(pf1),
                               .pf2(pf2),
                               .clkp(clkp),
                               .ref_bar(ref_bar),
                               .cntd(cntd),
                               .pf(pf));

initial begin
  clock = 1'bz;
  cycle_count = 0;
  rsyn = 0;
  d0 = 0;
  d1 = 0;
  d2 = 0;
  d3 = 0;
  d4 = 0;
  d5 = 0;
  d6 = 0;
  d7 = 0;
  pf0 = 0;
  pf1 = 0;
  pf2 = 0;
  ref_bar = 1;
end

always #100 begin
  if (clock) begin
    clock = 1'bz;
  end else begin
    clock = 1;
    cycle_count = cycle_count + 1;
  end
end

always @(posedge clock) begin
  #5
  // First scan line we just set all of the playfield data bits to zero.
  if (cycle_count == 9) begin
    pf0 = 1;
    pf1 = 0;
    pf2 = 0;
  end else if (cycle_count == 18) begin
    pf0 = 0;
    pf1 = 1;
    pf2 = 0;
  end else if (cycle_count == 27) begin
    pf0 = 0;
    pf1 = 0;
    pf2 = 1;
  end else if (cycle_count == 36) begin
    pf0 = 0;
    pf1 = 0;
    pf2 = 0;
  end else if (cycle_count >= 68 && cycle_count < 228) begin
    if (pf != 0) begin
      $display("ERROR pf != 0, cycle_count: %d, pf: %d", cycle_count, pf);
      $finish;
    end

  // Second scan line we set a pattern and check for correct output from pf.
  end else if (cycle_count == 228 + 0) begin
    d0 = 1;
    d1 = 0;
    d2 = 1;
    d3 = 0;
    d4 = 1;
    d5 = 0;
    d6 = 1;
    d7 = 0;
    pf0 = 1;
    pf1 = 0;
    pf2 = 0;
  end else if (cycle_count == 228 + 9) begin
    pf0 = 0;
    pf1 = 1;
    pf2 = 0;
  end else if (cycle_count == 228 + 18) begin
    pf0 = 0;
    pf1 = 0;
    pf2 = 1;
  end else if (cycle_count == 228 + 27) begin
    pf0 = 0;
    pf1 = 0;
    pf2 = 0;
  end else if ((cycle_count >= 228 + 68) && (cycle_count < 228 + 72) ||
               (cycle_count >= 228 + 76) && (cycle_count < 228 + 80)) begin
    if (pf != 1) begin
      $display("ERROR PF0 pf != 1, cycle_count: %d, pf: %d", cycle_count, pf);
      $finish;
    end
  end else if ((cycle_count >= 228 + 72) && (cycle_count < 228 + 76) ||
               (cycle_count >= 228 + 80) && (cycle_count < 228 + 84)) begin
    if (pf != 0) begin
      $display("ERROR PF0 pf != 0, cycle_count: %d, pf: %d", cycle_count, pf);
      $finish;
    end

  end else if (cycle_count == 684) begin
    $display("OK");
    $finish;
  end
end

endmodule  // tia_playfield_registers_sim

