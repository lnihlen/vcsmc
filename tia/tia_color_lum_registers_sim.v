`include "tia_color_lum_registers.v"

module tia_color_lum_registers_sim();

reg p0, m0, p1, m1, pf, bl, blank, cntd, score_bar, pfp_bar;
reg[7:1] d;
reg bkci, pfci, p1ci, p0ci, clkp;
wire d1, d2, d3, d4, d5, d6, d7;
wire blk_bar;
wire[7:1] o;
wire l0, l1, l2, c0, c1, c2, c3;

assign o[1] = l0;
assign o[2] = l1;
assign o[3] = l2;
assign o[4] = c0;
assign o[5] = c1;
assign o[6] = c2;
assign o[7] = c3;

integer cc;

tia_color_lum_registers clreg(.p0(p0), .m0(m0), .p1(p1), .m1(m1), .pf(pf),
    .bl(bl), .blank(blank), .cntd(cntd), .score_bar(score_bar),
    .pfp_bar(pfp_bar), .d1(d1), .d2(d2), .d3(d3), .d4(d4), .d5(d5), .d6(d6),
    .d7(d7), .bkci(bkci), .pfci(pfci), .p1ci(p1ci), .p0ci(p0ci), .clkp(clkp),
    .blk_bar(blk_bar), .l0(l0), .l1(l1), .l2(l2), .c0(c0), .c1(c1), .c2(c2),
    .c3(c3));

initial begin
  p0 = 0;
  m0 = 0;
  p1 = 0;
  m1 = 0;
  pf = 0;
  bl = 0;
  blank = 0;
  cntd = 0;
  score_bar = 1;
  pfp_bar = 1;
  d = 7'b1110000;
  bkci = 0;
  pfci = 0;
  p1ci = 0;
  p0ci = 0;
  clkp = 0;

  cc = 0;
end

assign d1 = d[1];
assign d2 = d[2];
assign d3 = d[3];
assign d4 = d[4];
assign d5 = d[5];
assign d6 = d[6];
assign d7 = d[7];

always #100 begin
  clkp = ~clkp;
end

always @(posedge clkp) begin
  #5
  case (cc)
    0: begin
      bkci = 1;
    end
    1: begin
      bkci = 0;
      #1
      d = 7'b1111111;
      pfci = 1;
    end
    2: begin
      pfci = 0;
      #1
      d = 7'b0101010;
      p1ci = 1;
    end
    3: begin
      p1ci = 0;
      #1
      d = 7'b1010101;
      p0ci = 1;
    end
    4: begin
      p0ci = 0;
      #1
      d = 7'b0000000;
    end
    5: begin
      if (o != 7'b1110000) begin
        $display("all signals at zero, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p0 = 1;
    end
    6: begin
      if (o != 7'b1010101) begin
        $display("p0 at one, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p0 = 0;
      p1 = 1;
    end
    7: begin
      if (o != 7'b0101010) begin
        $display("p1 at one, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p1 = 0;
      pf = 1;
    end
    8: begin
      if (o != 7'b1111111) begin
        $display("pf at one, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      pf = 0;
      m1 = 1;
    end
    9: begin
      if (o != 7'b0101010) begin
        $display("m1 at one, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      m1 = 0;
      m0 = 1;
    end
    10: begin
      if (o != 7'b1010101) begin
        $display("m0 at one, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      m0 = 0;
      bl = 1;
    end
    11: begin
      if (o != 7'b1111111) begin
        $display("bl at one, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p0 = 1;
      m0 = 1;
      p1 = 1;
      m1 = 1;
      pf = 1;
      bl = 1;
    end
    12: begin
      if (o != 7'b1010101) begin
        $display("p0 priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p0 = 0;
    end
    13: begin
      if (o != 7'b1010101) begin
        $display("m0 priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      m0 = 0;
    end
    14: begin
      if (o != 7'b0101010) begin
        $display("p1 priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p1 = 0;
    end
    15: begin
      if (o != 7'b0101010) begin
        $display("m1 priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      m1 = 0;
    end
    16: begin
      if (o != 7'b1111111) begin
        $display("pf priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      pf = 0;
    end
    17: begin
      if (o != 7'b1111111) begin
        $display("bl priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      bl = 0;
    end
    18: begin
      if (o != 7'b1110000) begin
        $display("all signals at zero, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      pfp_bar = 0;
      p0 = 1;
      m0 = 1;
      p1 = 1;
      m1 = 1;
      pf = 1;
      bl = 1;
    end
    19: begin
      if (o != 7'b1111111) begin
        $display("bl priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      bl = 0;
    end
    20: begin
      if (o != 7'b1111111) begin
        $display("pf priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      pf = 0;
    end
    21: begin
      if (o != 7'b1010101) begin
        $display("m0 priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      m0 = 0;
    end
    22: begin
      if (o != 7'b1010101) begin
        $display("p0 priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p0 = 0;
    end
    23: begin
      if (o != 7'b0101010) begin
        $display("m1 priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      m1 = 0;
    end
    24: begin
      if (o != 7'b0101010) begin
        $display("p1 priority, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p1 = 0;
      pfp_bar = 1;
      score_bar = 0;
    end
    25: begin
      if (o != 7'b1110000) begin
        $display("all signals at zero, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      pf = 1;
    end
    26: begin
      if (o != 7'b1010101) begin
        $display("score mode left side, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      cntd = 1;
    end
    27: begin
      if (o != 7'b0101010) begin
        $display("score mode right side, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p1 = 1;
    end
    28: begin
      if (o != 7'b0101010) begin
        $display("score mode p1, o: %b, cc: %d", o, cc);
        $finish;
      end
      #1
      p0 = 1;
    end
    29: begin
      if (o != 7'b1010101) begin
        $display("score mode p1, o: %b, cc: %d", o, cc);
        $finish;
      end
    end
    30: begin
      $display("OK");
      $finish;
    end
  endcase
  cc = cc + 1;
end

endmodule  // tia_color_lum_registers_sim
