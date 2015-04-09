`include "tia_l.v"

module tia_player_graphics_register_cell(d, gs, p0, p0_bar, p1, p1_bar, n, o);
  input d, gs, p0, p0_bar, p1, p1_bar;
  output n, o;
  wire d, gs, p0, p0_bar, p1, p1_bar, n, o;

  wire l_new_out, l_old_out;
  tia_l l_new(.in(d), .follow(p0), .latch(p0_bar), .out(l_new_out));
  tia_l l_old(.in(l_new_out), .follow(p1_bar), .latch(p1), .out(l_old_out));

  // New and old are 0 if the graphics bit is 1. Here the output is 1 if the
  // graphics bit is 1.
  assign n = gs & l_new_out;
  assign o = gs & l_old_out;
endmodule  // tia_player_graphics_register_cell

module tia_player_graphics_register(
    // input
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    gs0,
    gs1,
    gs2,
    p0gr,
    p1gr,
    // output
    new,
    old);

input d0, d1, d2, d3, d4, d5, d6, d7, gs0, gs1, gs2, p0gr, p1gr;
output new, old;

wire d0, d1, d2, d3, d4, d5, d6, d7, gs0, gs1, gs2, p0gr, p1gr;
wire new, old;

wire gs0_bar, gs1_bar, gs2_bar;
assign gs0_bar = ~gs0;
assign gs1_bar = ~gs1;
assign gs2_bar = ~gs2;

wire p0gr_bar, p1gr_bar;
assign p0gr_bar = ~p0gr;
assign p1gr_bar = ~p1gr;

wire g0, g1, g2, g3, g4, g5, g6, g7;
assign g0 = gs0_bar & gs1_bar & gs2_bar;
assign g1 = gs0     & gs1_bar & gs2_bar;
assign g2 = gs0_bar & gs1     & gs2_bar;
assign g3 = gs0     & gs1     & gs2_bar;
assign g4 = gs0_bar & gs1_bar & gs2;
assign g5 = gs0     & gs1_bar & gs2;
assign g6 = gs0_bar & gs1     & gs2;
assign g7 = gs0     & gs1     & gs2;

wire n0, n1, n2, n3, n4, n5, n6, n7;
wire o0, o1, o2, o3, o4, o5, o6, o7;

tia_player_graphics_register_cell c0(.d(d0), .gs(g0), .p0(p0gr),
    .p0_bar(p0gr_bar), .p1(p1gr), .p1_bar(p1gr_bar), .n(n0), .o(o0));
tia_player_graphics_register_cell c1(.d(d1), .gs(g1), .p0(p0gr),
    .p0_bar(p0gr_bar), .p1(p1gr), .p1_bar(p1gr_bar), .n(n1), .o(o1));
tia_player_graphics_register_cell c2(.d(d2), .gs(g2), .p0(p0gr),
    .p0_bar(p0gr_bar), .p1(p1gr), .p1_bar(p1gr_bar), .n(n2), .o(o2));
tia_player_graphics_register_cell c3(.d(d3), .gs(g3), .p0(p0gr),
    .p0_bar(p0gr_bar), .p1(p1gr), .p1_bar(p1gr_bar), .n(n3), .o(o3));
tia_player_graphics_register_cell c4(.d(d4), .gs(g4), .p0(p0gr),
    .p0_bar(p0gr_bar), .p1(p1gr), .p1_bar(p1gr_bar), .n(n4), .o(o4));
tia_player_graphics_register_cell c5(.d(d5), .gs(g5), .p0(p0gr),
    .p0_bar(p0gr_bar), .p1(p1gr), .p1_bar(p1gr_bar), .n(n5), .o(o5));
tia_player_graphics_register_cell c6(.d(d6), .gs(g6), .p0(p0gr),
    .p0_bar(p0gr_bar), .p1(p1gr), .p1_bar(p1gr_bar), .n(n6), .o(o6));
tia_player_graphics_register_cell c7(.d(d7), .gs(g7), .p0(p0gr),
    .p0_bar(p0gr_bar), .p1(p1gr), .p1_bar(p1gr_bar), .n(n7), .o(o7));

assign new = ~(n0 | n1 | n2 | n3 | n4 | n5 | n6 | n7);
assign old = ~(o0 | o1 | o2 | o3 | o4 | o5 | o6 | o7);

endmodule  // tia_player_graphics_register
