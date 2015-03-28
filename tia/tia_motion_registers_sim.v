`include "tia_motion_registers.v"
`include "tia_biphase_clock.v"

module tia_motion_registers_cell_a_sim();

reg clock;
reg rsyn;
wire hphi1;
wire hphi2;
wire rsynl;
wire out;
reg d4, d5, d6, d7;
reg p0hm, p1hm, m0hm, m1hm, blhm;
reg hmclr;
reg sec;
integer cc;
integer line_count;
integer p0ec_count, p1ec_count, m0ec_count, m1ec_count, blec_count;
reg[3:0] p0d;
reg[3:0] p1d;
reg[3:0] m0d;
reg[3:0] m1d;
reg[3:0] bld;
wire p0ec_bar, p1ec_bar, m0ec_bar, m1ec_bar, blec_bar;

tia_biphase_clock bpc(.clk(clock),
                      .rsyn(rsyn),
                      .hphi1(hphi1),
                      .hphi2(hphi2),
                      .rsynl(rsynl));

tia_motion_registers mr(.d4(d4),
                        .d5(d5),
                        .d6(d6),
                        .d7(d7),
                        .hmclr(hmclr),
                        .p0hm(p0hm),
                        .p1hm(p1hm),
                        .m0hm(m0hm),
                        .m1hm(m1hm),
                        .blhm(blhm),
                        .sec(sec),
                        .hphi1(hphi1),
                        .hphi2(hphi2),
                        .p0ec_bar(p0ec_bar),
                        .p1ec_bar(p1ec_bar),
                        .m0ec_bar(m0ec_bar),
                        .m1ec_bar(m1ec_bar),
                        .blec_bar(blec_bar));

initial begin
  clock = 1;
  rsyn = 0;
  sec = 0;
  cc = 0;
  line_count = 0;
  p0ec_count = 0;
  p1ec_count = 0;
  m0ec_count = 0;
  m1ec_count = 0;
  blec_count = 0;
  p0d = 4'b0111;
  p1d = 4'b0000;
  m0d = 4'b0100;
  m1d = 4'b1100;
  bld = 4'b0010;
  d4 = 0;
  d5 = 0;
  d6 = 0;
  d7 = 0;
  hmclr = 0;
  p0hm = 1;
  p1hm = 1;
  m0hm = 1;
  m1hm = 1;
  blhm = 1;
end

always #100 begin
  if (clock) begin
    clock = 1'bz;
  end else begin
    clock = 1;
  end
end

always @(posedge clock) begin
  if (line_count == 3) begin
    if (cc == 0) begin
      hmclr = 1;
    end else if (cc == 9) begin
      hmclr = 0;
    end
  end else begin
    if (cc == 0) begin
      p0hm = 1;
      p1hm = 0;
      m0hm = 0;
      m1hm = 0;
      blhm = 0;
      #1
      d4 = p0d[0];
      d5 = p0d[1];
      d6 = p0d[2];
      d7 = p0d[3];
    end else if (cc == 1) begin
      p0hm = 0;
      p1hm = 1;
      m0hm = 0;
      m1hm = 0;
      blhm = 0;
      #1
      d4 = p1d[0];
      d5 = p1d[1];
      d6 = p1d[2];
      d7 = p1d[3];
    end else if (cc == 2) begin
      p0hm = 0;
      p1hm = 0;
      m0hm = 1;
      m1hm = 0;
      blhm = 0;
      #1
      d4 = m0d[0];
      d5 = m0d[1];
      d6 = m0d[2];
      d7 = m0d[3];
    end else if (cc == 3) begin
      p0hm = 0;
      p1hm = 0;
      m0hm = 0;
      m1hm = 1;
      blhm = 0;
      #1
      d4 = m1d[0];
      d5 = m1d[1];
      d6 = m1d[2];
      d7 = m1d[3];
    end else if (cc == 4) begin
      p0hm = 0;
      p1hm = 0;
      m0hm = 0;
      m1hm = 0;
      blhm = 1;
      #1
      d4 = bld[0];
      d5 = bld[1];
      d6 = bld[2];
      d7 = bld[3];
    end else if (cc == 5) begin
      blhm = 0;
      #1
      d4 = 0;
      d5 = 0;
      d6 = 0;
      d7 = 0;
    end
  end
  if (cc == 9) begin
    sec = 1;
  end else if (cc == 13) begin
    sec = 0;
  end

  if (p0ec_bar === 0) begin
    p0ec_count = p0ec_count + 1;
  end
  if (p1ec_bar === 0) begin
    p1ec_count = p1ec_count + 1;
  end
  if (m0ec_bar === 0) begin
    m0ec_count = m0ec_count + 1;
  end
  if (m1ec_bar === 0) begin
    m1ec_count = m1ec_count + 1;
  end
  if (blec_bar === 0) begin
    blec_count = blec_count + 1;
  end

  if (cc == 228) begin
    if (line_count == 0) begin
      if (p0ec_count != 15) begin
        $display("p0d: %b, p0ec_count: %d", p0d, p0ec_count);
        $finish;
      end
      if (p1ec_count != 8) begin
        $display("p1d: %b, p1ec_count: %d", p1d, p1ec_count);
        $finish;
      end
      if (m0ec_count != 12) begin
        $display("m0d: %b, m0ec_count: %d", m0d, m0ec_count);
        $finish;
      end
      if (m1ec_count != 4) begin
        $display("m1d: %b, m1ec_count: %d", m1d, m1ec_count);
        $finish;
      end
      if (blec_count != 10) begin
        $display("bld: %b, blec_count: %d", bld, blec_count);
        $finish;
      end
      #1
      p0d = 4'b1000;
      p1d = 4'b1010;
      m0d = 4'b1111;
      m1d = 4'b0011;
      bld = 4'b1011;
    end else if (line_count == 1) begin
      if (p0ec_count != 0) begin
        $display("p0d: %b, p0ec_count: %d", p0d, p0ec_count);
        $finish;
      end
      if (p1ec_count != 2) begin
        $display("p1d: %b, p1ec_count: %d", p1d, p1ec_count);
        $finish;
      end
      if (m0ec_count != 7) begin
        $display("m0d: %b, m0ec_count: %d", m0d, m0ec_count);
        $finish;
      end
      if (m1ec_count != 11) begin
        $display("m1d: %b, m1ec_count: %d", m1d, m1ec_count);
        $finish;
      end
      if (blec_count != 3) begin
        $display("bld: %b, blec_count: %d", bld, blec_count);
        $finish;
      end
      #1
      p0d = 4'b0101;
      p1d = 4'b0001;
      m0d = 4'b1110;
      m1d = 4'b1101;
      bld = 4'b1001;
    end else if (line_count == 2) begin
      if (p0ec_count != 13) begin
        $display("p0d: %b, p0ec_count: %d", p0d, p0ec_count);
        $finish;
      end
      if (p1ec_count != 9) begin
        $display("p1d: %b, p1ec_count: %d", p1d, p1ec_count);
        $finish;
      end
      if (m0ec_count != 6) begin
        $display("m0d: %b, m0ec_count: %d", m0d, m0ec_count);
        $finish;
      end
      if (m1ec_count != 5) begin
        $display("m1d: %b, m1ec_count: %d", m1d, m1ec_count);
        $finish;
      end
      if (blec_count != 1) begin
        $display("bld: %b, blec_count: %d", bld, blec_count);
        $finish;
      end
      $display("OK");
      $finish;
    end else if (line_count == 3) begin
      if (p0ec_count != 8) begin
        $display("HMCLR p0d: %b, p0ec_count: %d", p0d, p0ec_count);
        $finish;
      end
      if (p1ec_count != 8) begin
        $display("HMCLR p1d: %b, p1ec_count: %d", p1d, p1ec_count);
        $finish;
      end
      if (m0ec_count != 8) begin
        $display("HMCLR m0d: %b, m0ec_count: %d", m0d, m0ec_count);
        $finish;
      end
      if (m1ec_count != 8) begin
        $display("HMCLR m1d: %b, m1ec_count: %d", m1d, m1ec_count);
        $finish;
      end
      if (blec_count != 8) begin
        $display("HMCLR bld: %b, blec_count: %d", bld, blec_count);
        $finish;
      end
      $display("OK");
      $finish;
    end
    #1
    p0ec_count = 0;
    p1ec_count = 0;
    m0ec_count = 0;
    m1ec_count = 0;
    blec_count = 0;
    cc = 0;
    line_count = line_count + 1;
  end else begin
    cc = cc + 1;
  end
end

endmodule  // tia_motion_registers_cell_a
