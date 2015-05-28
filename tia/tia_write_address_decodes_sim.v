`include "tia_write_address_decodes.v"

module tia_write_address_decodes_sim();

reg[5:0] a;
reg phi2;
reg w_bar;
wire vsyn, vblk, wsyn, rsyn, nsz0, nsz1, p0ci, p1ci, pfci, bkci, pfct, p0rf,
  p1rf, pf0, pf1, pf2, p0re, p1re, m0re, m1re, blre, auc0, auc1, auf0, auf1,
  auv0, auv1, p0gr, p1gr, m0en, m1en, blen, p0hm, p1hm, m0hm, m1hm, blhm, p0vd,
  p1vd, blvd, m0pre, m1pre, hmove, hmclr, cxclr;

integer oc;

tia_write_address_decodes decode(
  .a(a),
  .phi2(phi2),
  .w_bar(w_bar),
  .vsyn(vsyn),
  .vblk(vblk),
  .wsyn(wsyn),
  .rsyn(rsyn),
  .nsz0(nsz0),
  .nsz1(nsz1),
  .p0ci(p0ci),
  .p1ci(p1ci),
  .pfci(pfci),
  .bkci(bkci),
  .pfct(pfct),
  .p0rf(p0rf),
  .p1rf(p1rf),
  .pf0(pf0),
  .pf1(pf1),
  .pf2(pf2),
  .p0re(p0re),
  .p1re(p1re),
  .m0re(m0re),
  .m1re(m1re),
  .blre(blre),
  .auc0(auc0),
  .auc1(auc1),
  .auf0(auf0),
  .auf1(auf1),
  .auv0(auv0),
  .auv1(auv1),
  .p0gr(p0gr),
  .p1gr(p1gr),
  .m0en(m0en),
  .m1en(m1en),
  .blen(blen),
  .p0hm(p0hm),
  .p1hm(p1hm),
  .m0hm(m0hm),
  .m1hm(m1hm),
  .blhm(blhm),
  .p0vd(p0vd),
  .p1vd(p1vd),
  .blvd(blvd),
  .m0pre(m0pre),
  .m1pre(m1pre),
  .hmove(hmove),
  .hmclr(hmclr),
  .cxclr(cxclr));

initial begin
  a = 6'b000000;
  phi2 = 0;
  w_bar = 0;

  $dumpfile("out/tia_write_address_decodes_sim.vcd");
  $dumpvars(0, tia_write_address_decodes_sim);
end

always #100 begin
  phi2 = ~phi2;
  if (phi2) a = a + 1;
  if (a > 44) begin
    $display("OK");
    $finish;
  end
end

always @(negedge phi2) begin
  oc = 0;
  #1
  // It turns out that detecting the existence of only a single 1 in an array
  // of bits is a very interesting digital logic problem. Which we skip here
  // entirely and just do something obvious :).
  if (vsyn ) oc = oc + 1;
  if (vblk ) oc = oc + 1;
  if (wsyn ) oc = oc + 1;
  if (rsyn ) oc = oc + 1;
  if (nsz0 ) oc = oc + 1;
  if (nsz1 ) oc = oc + 1;
  if (p0ci ) oc = oc + 1;
  if (p1ci ) oc = oc + 1;
  if (pfci ) oc = oc + 1;
  if (bkci ) oc = oc + 1;
  if (pfct ) oc = oc + 1;
  if (p0rf ) oc = oc + 1;
  if (p1rf ) oc = oc + 1;
  if (pf0  ) oc = oc + 1;
  if (pf1  ) oc = oc + 1;
  if (pf2  ) oc = oc + 1;
  if (p0re ) oc = oc + 1;
  if (p1re ) oc = oc + 1;
  if (m0re ) oc = oc + 1;
  if (m1re ) oc = oc + 1;
  if (blre ) oc = oc + 1;
  if (auc0 ) oc = oc + 1;
  if (auc1 ) oc = oc + 1;
  if (auf0 ) oc = oc + 1;
  if (auf1 ) oc = oc + 1;
  if (auv0 ) oc = oc + 1;
  if (auv1 ) oc = oc + 1;
  if (p0gr ) oc = oc + 1;
  if (p1gr ) oc = oc + 1;
  if (m0en ) oc = oc + 1;
  if (m1en ) oc = oc + 1;
  if (blen ) oc = oc + 1;
  if (p0hm ) oc = oc + 1;
  if (p1hm ) oc = oc + 1;
  if (m0hm ) oc = oc + 1;
  if (m1hm ) oc = oc + 1;
  if (blhm ) oc = oc + 1;
  if (p0vd ) oc = oc + 1;
  if (p1vd ) oc = oc + 1;
  if (blvd ) oc = oc + 1;
  if (m0pre) oc = oc + 1;
  if (m1pre) oc = oc + 1;
  if (hmove) oc = oc + 1;
  if (hmclr) oc = oc + 1;
  if (cxclr) oc = oc + 1;
  #1
  if (oc != 1) begin
    $display("%d ones detected on address %b", oc, a);
    $finish;
  end
  case (a)
    0: begin
      if (vsyn != 1) begin
        $display("vsyn: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    1: begin
      if (vblk != 1) begin
        $display("vblk: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    2: begin
      if (wsyn != 1) begin
        $display("wsyn: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    3: begin
      if (rsyn != 1) begin
        $display("rsyn: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    4: begin
      if (nsz0 != 1) begin
        $display("nsz0: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    5: begin
      if (nsz1 != 1) begin
        $display("nsz1: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    6: begin
      if (p0ci != 1) begin
        $display("p0ci: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    7: begin
      if (p1ci != 1) begin
        $display("p1ci: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    8: begin
      if (pfci != 1) begin
        $display("pfci: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    9: begin
      if (bkci != 1) begin
        $display("bkci: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    10: begin
      if (pfct != 1) begin
        $display("pfct: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    11: begin
      if (p0rf != 1) begin
        $display("p0rf: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    12: begin
      if (p1rf != 1) begin
        $display("p1rf: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    13: begin
      if (pf0 != 1) begin
        $display("pf0: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    14: begin
      if (pf1 != 1) begin
        $display("pf1: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    15: begin
      if (pf2 != 1) begin
        $display("pf2: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    16: begin
      if (p0re != 1) begin
        $display("p0re: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    17: begin
      if (p1re != 1) begin
        $display("p1re: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    18: begin
      if (m0re != 1) begin
        $display("m0re: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    19: begin
      if (m1re != 1) begin
        $display("m1re: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    20: begin
      if (blre != 1) begin
        $display("blre: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    21: begin
      if (auc0 != 1) begin
        $display("auc0: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    22: begin
      if (auc1 != 1) begin
        $display("auc1: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    23: begin
      if (auf0 != 1) begin
        $display("auf0: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    24: begin
      if (auf1 != 1) begin
        $display("auf1: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    25: begin
      if (auv0 != 1) begin
        $display("auv0: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    26: begin
      if (auv1 != 1) begin
        $display("auv1: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    27: begin
      if (p0gr != 1) begin
        $display("p0gr: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    28: begin
      if (p1gr != 1) begin
        $display("p1gr: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    29: begin
      if (m0en != 1) begin
        $display("m0en: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    30: begin
      if (m1en != 1) begin
        $display("m1en: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    31: begin
      if (blen != 1) begin
        $display("blen: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    32: begin
      if (p0hm != 1) begin
        $display("p0hm: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    33: begin
      if (p1hm != 1) begin
        $display("p1hm: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    34: begin
      if (m0hm != 1) begin
        $display("m0hm: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    35: begin
      if (m1hm != 1) begin
        $display("m1hm: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    36: begin
      if (blhm != 1) begin
        $display("blhm: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    37: begin
      if (p0vd != 1) begin
        $display("p0vd: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    38: begin
      if (p1vd != 1) begin
        $display("p1vd: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    39: begin
      if (blvd != 1) begin
        $display("blvd: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    40: begin
      if (m0pre != 1) begin
        $display("m0pre: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    41: begin
      if (m1pre != 1) begin
        $display("m1pre: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    42: begin
      if (hmove != 1) begin
        $display("hmove: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    43: begin
      if (hmclr != 1) begin
        $display("hmclr: %d, a: %b", vsyn, a);
        $finish;
      end
    end
    44: begin
      if (cxclr != 1) begin
        $display("cxclr: %d, a: %b", vsyn, a);
        $finish;
      end
    end
  endcase
end

endmodule  // tia_write_address_decodes_sim
