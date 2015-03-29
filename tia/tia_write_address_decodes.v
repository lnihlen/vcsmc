module tia_write_address_decodes(
  a,
  phi2,
  w_bar,
  vsyn,
  vblk,
  wsyn,
  rsyn,
  nsz0,
  nsz1,
  p0ci,
  p1ci,
  pfci,
  bkci,
  pfct,
  p0rf,
  p1rf,
  pf0,
  pf1,
  pf2,
  p0re,
  p1re,
  m0re,
  m1re,
  blre,
  auc0,
  auc1,
  auf0,
  auf1,
  auv0,
  auv1,
  p0cr,
  p1cr,
  m0en,
  m1en,
  blen,
  p0hm,
  p1hm,
  m0hm,
  m1hm,
  blhm,
  p0vd,
  p1vd,
  blvd,
  m0pre,
  m1pre,
  hmove,
  hmclr,
  cxclr);

input[5:0] a;
input phi2, w_bar;
output vsyn, vblk, wsyn, rsyn, nsz0, nsz1, p0ci, p1ci, pfci, bkci, pfct, p0rf,
  p1rf, pf0, pf1, pf2, p0re, p1re, m0re, m1re, blre, auc0, auc1, auf0, auf1,
  auv0, auv1, p0cr, p1cr, m0en, m1en, blen, p0hm, p1hm, m0hm, m1hm, blhm, p0vd,
  p1vd, blvd, m0pre, m1pre, hmove, hmclr, cxclr;

wire[5:0] a;
wire phi2, w_bar;
wire vsyn, vblk, wsyn, rsyn, nsz0, nsz1, p0ci, p1ci, pfci, bkci, pfct, p0rf,
  p1rf, pf0, pf1, pf2, p0re, p1re, m0re, m1re, blre, auc0, auc1, auf0, auf1,
  auv0, auv1, p0cr, p1cr, m0en, m1en, blen, p0hm, p1hm, m0hm, m1hm, blhm, p0vd,
  p1vd, blvd, m0pre, m1pre, hmove, hmclr, cxclr;


wire[5:0] n;
assign n = ~a;
wire p;
assign p = (~phi2) & (~w_bar);

assign vsyn  = p & n[5] & n[4] & n[3] & n[2] & n[1] & n[0];  // 00
assign vblk  = p & n[5] & n[4] & n[3] & n[2] & n[1] & a[0];  // 01
assign wsyn  = p & n[5] & n[4] & n[3] & n[2] & a[1] & n[0];  // 02
assign rsyn  = p & n[5] & n[4] & n[3] & n[2] & a[1] & a[0];  // 03
assign nsz0  = p & n[5] & n[4] & n[3] & a[2] & n[1] & n[0];  // 04
assign nsz1  = p & n[5] & n[4] & n[3] & a[2] & n[1] & a[0];  // 05
assign p0ci  = p & n[5] & n[4] & n[3] & a[2] & a[1] & n[0];  // 06
assign p1ci  = p & n[5] & n[4] & n[3] & a[2] & a[1] & a[0];  // 07
assign pfci  = p & n[5] & n[4] & a[3] & n[2] & n[1] & n[0];  // 08
assign bkci  = p & n[5] & n[4] & a[3] & n[2] & n[1] & a[0];  // 09
assign pfct  = p & n[5] & n[4] & a[3] & n[2] & a[1] & n[0];  // 0a
assign p0rf  = p & n[5] & n[4] & a[3] & n[2] & a[1] & a[0];  // 0b
assign p1rf  = p & n[5] & n[4] & a[3] & a[2] & n[1] & n[0];  // 0c
assign pf0   = p & n[5] & n[4] & a[3] & a[2] & n[1] & a[0];  // 0d
assign pf1   = p & n[5] & n[4] & a[3] & a[2] & a[1] & n[0];  // 0e
assign pf2   = p & n[5] & n[4] & a[3] & a[2] & a[1] & a[0];  // 0f
assign p0re  = p & n[5] & a[4] & n[3] & n[2] & n[1] & n[0];  // 10
assign p1re  = p & n[5] & a[4] & n[3] & n[2] & n[1] & a[0];  // 11
assign m0re  = p & n[5] & a[4] & n[3] & n[2] & a[1] & n[0];  // 12
assign m1re  = p & n[5] & a[4] & n[3] & n[2] & a[1] & a[0];  // 13
assign blre  = p & n[5] & a[4] & n[3] & a[2] & n[1] & n[0];  // 14
assign auc0  = p & n[5] & a[4] & n[3] & a[2] & n[1] & a[0];  // 15
assign auc1  = p & n[5] & a[4] & n[3] & a[2] & a[1] & n[0];  // 16
assign auf0  = p & n[5] & a[4] & n[3] & a[2] & a[1] & a[0];  // 17
assign auf1  = p & n[5] & a[4] & a[3] & n[2] & n[1] & n[0];  // 18
assign auv0  = p & n[5] & a[4] & a[3] & n[2] & n[1] & a[0];  // 19
assign auv1  = p & n[5] & a[4] & a[3] & n[2] & a[1] & n[0];  // 1a
assign p0cr  = p & n[5] & a[4] & a[3] & n[2] & a[1] & a[0];  // 1b
assign p1cr  = p & n[5] & a[4] & a[3] & a[2] & n[1] & n[0];  // 1c
assign m0en  = p & n[5] & a[4] & a[3] & a[2] & n[1] & a[0];  // 1d
assign m1en  = p & n[5] & a[4] & a[3] & a[2] & a[1] & n[0];  // 1e
assign blen  = p & n[5] & a[4] & a[3] & a[2] & a[1] & a[0];  // 1f
assign p0hm  = p & a[5] & n[4] & n[3] & n[2] & n[1] & n[0];  // 20
assign p1hm  = p & a[5] & n[4] & n[3] & n[2] & n[1] & a[0];  // 21
assign m0hm  = p & a[5] & n[4] & n[3] & n[2] & a[1] & n[0];  // 22
assign m1hm  = p & a[5] & n[4] & n[3] & n[2] & a[1] & a[0];  // 23
assign blhm  = p & a[5] & n[4] & n[3] & a[2] & n[1] & n[0];  // 24
assign p0vd  = p & a[5] & n[4] & n[3] & a[2] & n[1] & a[0];  // 25
assign p1vd  = p & a[5] & n[4] & n[3] & a[2] & a[1] & n[0];  // 26
assign blvd  = p & a[5] & n[4] & n[3] & a[2] & a[1] & a[0];  // 27
assign m0pre = p & a[5] & n[4] & a[3] & n[2] & n[1] & n[0];  // 28
assign m1pre = p & a[5] & n[4] & a[3] & n[2] & n[1] & a[0];  // 29
assign hmove = p & a[5] & n[4] & a[3] & n[2] & a[1] & n[0];  // 2a
assign hmclr = p & a[5] & n[4] & a[3] & n[2] & a[1] & a[0];  // 2b
assign cxclr = p & a[5] & n[4] & a[3] & a[2] & n[1] & n[0];  // 2c

endmodule  // tia_write_address_decodes
