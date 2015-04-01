`include "tia_f1.v"
`include "tia_f3.v"

module tia_player_graphics_scan_counter(
  // input
  start_bar,
  fstob,
  pck,
  count_bar,
  new,
  old,
  player_vert_delay_bar,
  missile_to_player_reset_bar,
  player_reflect_bar,
  clkp,
  // output
  missile_to_player_reset,
  gs0,
  gs1,
  gs2,
  p);

input start_bar, fstob, pck, count_bar, new, old, player_vert_delay_bar,
    missile_to_player_reset_bar, player_reflect_bar, clkp;
output missile_to_player_reset, gs0, gs1, gs2, p;

wire count, ena_bar;
assign count = ~count_bar;
tia_f1 cntf1(.s(count_bar), .r(count), .clock(pck), .reset(0), .q_bar(ena_bar));

wire stop, f3ar, f3aq_bar;
assign f3ar = ~(start_bar & stop);
tia_f3 f3a(.s1(start_bar), .s2(ena_bar), .r1(ena_bar), .r2(f3ar), .clock(pck),
    .reset(0), .q_bar(f3aq_bar));

wire ar, f3bq, f3bq_bar;
assign ar = ena_bar | f3aq_bar;
tia_f3 f3b(.s1(f3bq), .s2(ar), .r1(ar), .r2(f3bq_bar), .clock(pck), .reset(0),
    .q(f3bq), .q_bar(f3bq_bar));

wire br, f3cq, f3cq_bar;
assign br = ar | f3bq_bar;
tia_f3 f3c(.s1(f3cq), .s2(br), .r1(br), .r2(f3cq_bar), .clock(pck), .reset(0),
    .q(f3cq), .q_bar(f3cq_bar));

wire cr, f3dq, f3dq_bar;
assign cr = br | f3cq_bar;
tia_f3 f3d(.s1(f3dq), .s2(cr), .r1(cr), .r2(f3dq_bar), .clock(pck), .reset(0),
    .q(f3dq), .q_bar(f3dq_bar));

assign missile_to_player_reset = ~(fstob | ena_bar | pck | f3dq | f3cq |
      f3bq_bar | missile_to_player_reset_bar);
assign stop = ~(f3bq_bar | f3cq_bar | f3dq_bar);

wire pf2s, pf2r;
assign pf2s = ~((~(f3aq_bar | old | player_vert_delay_bar)) |
    (~((~player_vert_delay_bar) | new | f3aq_bar)));
assign pf2r = ~pf2s;
tia_f1 pf1(.s(pf2s), .r(pf2r), .clock(clkp), .reset(0), .q(p));

wire player_reflect;
assign player_reflect = ~player_reflect_bar;

assign gs0 = ~((player_reflect & f3bq_bar) | (f3bq & player_reflect_bar));
assign gs1 = ~((player_reflect & f3cq_bar) | (f3cq & player_reflect_bar));
assign gs2 = ~((player_reflect & f3dq_bar) | (f3dq & player_reflect_bar));

endmodule  // tia_player_graphics_scan_counter
