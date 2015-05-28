`include "tia_player_graphics.v"

module tia_player_graphics_sim();

reg motck, pxec_bar, pxre, clkp, mxre, mxpre, pxvd, mxen, nszx, mxec_bar;
reg pxgr, pygr;
reg[7:0] d;

wire d0, d1, d2, d3, d4, d5, d6, d7, px, mx;
assign d0 = d[0];
assign d1 = d[1];
assign d2 = d[2];
assign d3 = d[3];
assign d4 = d[4];
assign d5 = d[5];
assign d6 = d[6];
assign d7 = d[7];

tia_player_graphics tpg(.motck(motck),
                        .pxec_bar(pxec_bar),
                        .pxre(pxre),
                        .clkp(clkp),
                        .mxre(mxre),
                        .mxpre(mxpre),
                        .pxvd(pxvd),
                        .mxen(mxen),
                        .nszx(nszx),
                        .mxec_bar(mxec_bar),
                        .pxgr(pxgr),
                        .pygr(pygr),
                        .d0(d0),
                        .d1(d1),
                        .d2(d2),
                        .d3(d3),
                        .d4(d4),
                        .d5(d5),
                        .d6(d6),
                        .d7(d7),
                        .px(px),
                        .mx(mx));

initial begin
  motck = 0;
  pxec_bar = 1;
  pxre = 0;
  clkp = 0;
  mxre = 0;
  mxpre = 0;
  pxvd = 0;
  mxen = 0;
  nszx = 0;
  mxec_bar = 1;
  pxgr = 0;
  pygr = 0;
  d = 8'b00000000;

  $dumpfile("out/tia_player_graphics_sim.vcd");
  $dumpvars(0, tia_player_graphics_sim);
end

endmodule  // tia_player_graphics_sim
