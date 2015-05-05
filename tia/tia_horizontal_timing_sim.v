`include "tia_biphase_clock.v"
`include "tia_horizontal_lfsr.v"
`include "tia_horizontal_lfsr_decoder.v"
`include "tia_horizontal_timing.v"

module tia_horizontal_timing_sim();

reg clock;
reg rsyn;
wire hphi1;
wire hphi2;
wire rsynl;
wire[5:0] lfsr_out;
wire shb;
wire rsynd;

wire rhs;
wire cnt;
wire rcb;
wire shs;
wire lrhb;
wire rhb;

integer cc;

tia_biphase_clock bpc(.clk(clock),
                      .r(rsyn),
                      .phi1(hphi1),
                      .phi2(hphi2),
                      .rl(rsynl));

tia_horizontal_lfsr lfsr(.hphi1(hphi1),
                         .hphi2(hphi2),
                         .rsynl(rsynl),
                         .out(lfsr_out),
                         .shb(shb),
                         .rsynd(rsynd));

tia_horizontal_lfsr_decoder decode(.in(lfsr_out),
                                   .rhs(rhs),
                                   .cnt(cnt),
                                   .rcb(rcb),
                                   .shs(shs),
                                   .lrhb(lrhb),
                                   .rhb(rhb));

reg vsyn;
reg vblk;
reg hmove;
wire clkp;
assign clkp = ~clock;
reg wsyn;
reg d1;
wire resphi0;
wire aphi1, aphi2;
wire cb, vb, vb_bar, blank, motck, rdy, sec, syn;

tia_horizontal_timing timing(.hphi1(hphi1),
                             .hphi2(hphi2),
                             .rhs(rhs),
                             .cnt(cnt),
                             .rcb(rcb),
                             .lrhb(lrhb),
                             .rhb(rhb),
                             .shs(shs),
                             .vsyn(vsyn),
                             .vblk(vblk),
                             .rsynd(rsynd),
                             .hmove(hmove),
                             .clkp(clkp),
                             .wsyn(wsyn),
                             .d1(d1),
                             .clk(clock),
                             .shb(shb),
                             .resphi0(resphi0),
                             .aphi1(aphi1),
                             .aphi2(aphi2),
                             .cb(cb),
                             .vb(vb),
                             .vb_bar(vb_bar),
                             .blank(blank),
                             .motck(motck),
                             .rdy(rdy),
                             .sec(sec),
                             .syn(syn));

initial begin
  clock = 0;
  cc = 0;
  rsyn = 0;
  vsyn = 0;
  vblk = 0;
  hmove = 0;
  wsyn = 1;
  d1 = 0;
end

always #100 begin
  clock = ~clock;
end

always @(rdy) begin
  $display("%d: %d", cc, rdy);
end

// 228

always @(posedge clock) begin
  if (cc == 4) wsyn = 0;
  if (cc == 227) wsyn = 1;
  if (cc == 250) wsyn = 0;
  if (cc == 453) wsyn = 1;
  if (cc == 460) wsyn = 0;
  // TODO: actually test something here!
  cc = cc + 1;
  if (cc > 1000) begin
    $display("OK");
    $finish;
  end
end

endmodule  // tia_horizontal_timing_sim
