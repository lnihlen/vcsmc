`include "tia_biphase_clock.v"
`include "tia_horizontal_lfsr.v"

module tia_horizontal_lfsr_sim();

reg clock;
reg rsyn;
wire hphi1;
wire hphi2;
wire rsynl;
wire a;
wire b;
wire c;
wire d;
wire e;
wire f;
wire shb;
wire rsynd;

integer cycle_count;

tia_biphase_clock bpc(.clk(clock),
                      .rsyn(rsyn),
                      .hphi1(hphi1),
                      .hphi2(hphi2),
                      .rsynl(rsynl));

tia_horizontal_lfsr lfsr(.hphi1(hphi1),
                         .hphi2(hphi2),
                         .rsynl(rsynl),
                         .a(a),
                         .b(b),
                         .c(c),
                         .d(d),
                         .e(e),
                         .f(f),
                         .shb(shb),
                         .rsynd(rsynd));

initial begin
  clock = 1'bz;
  cycle_count = 0;
  rsyn = 0;
end

always #100 begin
  if (clock) begin
    clock = 1'bz;
  end else begin
    clock = 1;
  end
end

always @(posedge hphi1) begin
  #10
  $display("%d%d%d%d%d%d", a, b, c, d, e, f);
  if (cycle_count > 25) begin
    $display("OK");
    $finish;
  end
  cycle_count = cycle_count + 1;
end

endmodule  // tia_horizontal_lfsr_sim
