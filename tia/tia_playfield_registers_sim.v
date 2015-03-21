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

integer cc;
integer line_count;

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

reg [3:0] pf0_data;
reg [7:0] pf1_data;
reg [7:0] pf2_data;
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
  cc = 0;
  line_count = 0;
  rsyn = 0;
  pf0_data[3:0] = 4'b0000;
  pf1_data[7:0] = 8'b00000000;
  pf2_data[7:0] = 8'b00000000;
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
    cc = cc + 1;
  end
end

always @(posedge clkp) begin
  // We clock in the playfield registers during horizontal blanking
  if (cc == 1) begin
    d4 = pf0_data[0];
    d5 = pf0_data[1];
    d6 = pf0_data[2];
    d7 = pf0_data[3];
    #1
    pf0 = 1;
  end else if (cc == 2) begin
    d0 = pf1_data[0];
    d1 = pf1_data[1];
    d2 = pf1_data[2];
    d3 = pf1_data[3];
    d4 = pf1_data[4];
    d5 = pf1_data[5];
    d6 = pf1_data[6];
    d7 = pf1_data[7];
    #1
    pf0 = 0;
    pf1 = 1;
  end else if (cc == 3) begin
    d0 = pf2_data[0];
    d1 = pf2_data[1];
    d2 = pf2_data[2];
    d3 = pf2_data[3];
    d4 = pf2_data[4];
    d5 = pf2_data[5];
    d6 = pf2_data[6];
    d7 = pf2_data[7];
    #1
    pf1 = 0;
    pf2 = 1;
  end else if (cc == 4) begin
    pf2 = 0;
  end else if (cc >= 68 +  0 && cc < 68 +  4) begin // pf0 bit 4
    if (pf != pf0_data[0]) begin
      $display("pf0 bit 4 error! pf: %d, pf0_data[0]: %d", pf, pf0_data[0]);
      $finish;
    end
  end else if (cc >= 68 +  4 && cc < 68 +  8) begin // pf0 bit 5
    if (pf != pf0_data[1]) begin
      $display("pf0 bit 5 error! pf: %d, pf0_data[1]: %d", pf, pf0_data[1]);
      $finish;
    end
  end else if (cc >= 68 +  8 && cc < 68 + 12) begin // pf0 bit 6
    if (pf != pf0_data[2]) begin
      $display("pf0 bit 6 error! pf: %d, pf0_data[2]: %d", pf, pf0_data[2]);
      $finish;
    end
  end else if (cc >= 68 + 12 && cc < 68 + 16) begin // pf0 bit 7
    if (pf != pf0_data[3]) begin
      $display("pf0 bit 7 error! pf: %d, pf0_data[3]: %d", pf, pf0_data[3]);
      $finish;
    end
  end else if (cc >= 68 + 16 && cc < 68 + 20) begin // pf1 bit 7
    if (pf != pf1_data[7]) begin
      $display("pf1 bit 7 error! pf: %d, pf1_data[7]: %d", pf, pf1_data[7]);
      $finish;
    end
  end else if (cc >= 68 + 20 && cc < 68 + 24) begin // pf1 bit 6
    if (pf != pf1_data[6]) begin
      $display("pf1 bit 6 error! pf: %d, pf1_data[6]: %d", pf, pf1_data[6]);
      $finish;
    end
  end else if (cc >= 68 + 24 && cc < 68 + 28) begin // pf1 bit 5
    if (pf != pf1_data[5]) begin
      $display("pf1 bit 5 error! pf: %d, pf1_data[5]: %d", pf, pf1_data[5]);
      $finish;
    end
  end else if (cc >= 68 + 28 && cc < 68 + 32) begin // pf1 bit 4
    if (pf != pf1_data[4]) begin
      $display("pf1 bit 4 error! pf: %d, pf1_data[4]: %d", pf, pf1_data[4]);
      $finish;
    end
  end else if (cc >= 68 + 32 && cc < 68 + 36) begin // pf1 bit 3
    if (pf != pf1_data[3]) begin
      $display("pf1 bit 3 error! pf: %d, pf1_data[3]: %d", pf, pf1_data[3]);
      $finish;
    end
  end else if (cc >= 68 + 36 && cc < 68 + 40) begin // pf1 bit 2
    if (pf != pf1_data[2]) begin
      $display("pf1 bit 2 error! pf: %d, pf1_data[2]: %d", pf, pf1_data[2]);
      $finish;
    end
  end else if (cc >= 68 + 40 && cc < 68 + 44) begin // pf1 bit 1
    if (pf != pf1_data[1]) begin
      $display("pf1 bit 1 error! pf: %d, pf1_data[1]: %d", pf, pf1_data[1]);
      $finish;
    end
  end else if (cc >= 68 + 44 && cc < 68 + 48) begin // pf1 bit 0
    if (pf != pf1_data[0]) begin
      $display("pf1 bit 0 error! pf: %d, pf1_data[0]: %d", pf, pf1_data[0]);
      $finish;
    end
  end else if (cc >= 68 + 48 && cc < 68 + 52) begin // pf2 bit 0
    if (pf != pf2_data[0]) begin
      $display("pf2 bit 0 error! pf: %d, pf2_data[0]: %d", pf, pf2_data[0]);
      $finish;
    end
  end else if (cc >= 68 + 52 && cc < 68 + 56) begin // pf2 bit 1
    if (pf != pf2_data[1]) begin
      $display("pf2 bit 1 error! pf: %d, pf2_data[1]: %d", pf, pf2_data[1]);
      $finish;
    end
  end else if (cc >= 68 + 56 && cc < 68 + 60) begin // pf2 bit 2
    if (pf != pf2_data[2]) begin
      $display("pf2 bit 2 error! pf: %d, pf2_data[2]: %d", pf, pf2_data[2]);
      $finish;
    end
  end else if (cc >= 68 + 60 && cc < 68 + 64) begin // pf2 bit 3
    if (pf != pf2_data[3]) begin
      $display("pf2 bit 3 error! pf: %d, pf2_data[3]: %d", pf, pf2_data[3]);
      $finish;
    end
  end else if (cc >= 68 + 64 && cc < 68 + 68) begin // pf2 bit 4
    if (pf != pf2_data[4]) begin
      $display("pf2 bit 4 error! pf: %d, pf2_data[4]: %d", pf, pf2_data[4]);
      $finish;
    end
  end else if (cc >= 68 + 68 && cc < 68 + 72) begin // pf2 bit 5
    if (pf != pf2_data[5]) begin
      $display("pf2 bit 5 error! pf: %d, pf2_data[5]: %d", pf, pf2_data[5]);
      $finish;
    end
  end else if (cc >= 68 + 72 && cc < 68 + 76) begin // pf2 bit 6
    if (pf != pf2_data[6]) begin
      $display("pf2 bit 6 error! pf: %d, pf2_data[6]: %d", pf, pf2_data[6]);
      $finish;
    end
  end else if (cc >= 68 + 76 && cc < 68 + 80) begin // pf2 bit 7
    if (pf != pf2_data[7]) begin
      $display("pf2 bit 7 error! pf: %d, pf2_data[7]: %d", pf, pf2_data[7]);
      $finish;
    end
  end else if (cc >= 68 + 80 && cc < 68 + 160) begin
    if (ref_bar === 1) begin
      if (cc >= 148 +  0 && cc < 148 +  4) begin          // pf0 bit 4
        if (pf != pf0_data[0]) begin
          $display("pf0 bit 4 error! pf: %d, pf0_data[0]: %d", pf, pf0_data[0]);
          $finish;
        end
      end else if (cc >= 148 +  4 && cc < 148 +  8) begin // pf0 bit 5
        if (pf != pf0_data[1]) begin
          $display("pf0 bit 5 error! pf: %d, pf0_data[1]: %d", pf, pf0_data[1]);
          $finish;
        end
      end else if (cc >= 148 +  8 && cc < 148 + 12) begin // pf0 bit 6
        if (pf != pf0_data[2]) begin
          $display("pf0 bit 6 error! pf: %d, pf0_data[2]: %d", pf, pf0_data[2]);
          $finish;
        end
      end else if (cc >= 148 + 12 && cc < 148 + 16) begin // pf0 bit 7
        if (pf != pf0_data[3]) begin
          $display("pf0 bit 7 error! pf: %d, pf0_data[3]: %d", pf, pf0_data[3]);
          $finish;
        end
      end else if (cc >= 148 + 16 && cc < 148 + 20) begin // pf1 bit 7
        if (pf != pf1_data[7]) begin
          $display("pf1 bit 7 error! pf: %d, pf1_data[7]: %d", pf, pf1_data[7]);
          $finish;
        end
      end else if (cc >= 148 + 20 && cc < 148 + 24) begin // pf1 bit 6
        if (pf != pf1_data[6]) begin
          $display("pf1 bit 6 error! pf: %d, pf1_data[6]: %d", pf, pf1_data[6]);
          $finish;
        end
      end else if (cc >= 148 + 24 && cc < 148 + 28) begin // pf1 bit 5
        if (pf != pf1_data[5]) begin
          $display("pf1 bit 5 error! pf: %d, pf1_data[5]: %d", pf, pf1_data[5]);
          $finish;
        end
      end else if (cc >= 148 + 28 && cc < 148 + 32) begin // pf1 bit 4
        if (pf != pf1_data[4]) begin
          $display("pf1 bit 4 error! pf: %d, pf1_data[4]: %d", pf, pf1_data[4]);
          $finish;
        end
      end else if (cc >= 148 + 32 && cc < 148 + 36) begin // pf1 bit 3
        if (pf != pf1_data[3]) begin
          $display("pf1 bit 3 error! pf: %d, pf1_data[3]: %d", pf, pf1_data[3]);
          $finish;
        end
      end else if (cc >= 148 + 36 && cc < 148 + 40) begin // pf1 bit 2
        if (pf != pf1_data[2]) begin
          $display("pf1 bit 2 error! pf: %d, pf1_data[2]: %d", pf, pf1_data[2]);
          $finish;
        end
      end else if (cc >= 148 + 40 && cc < 148 + 44) begin // pf1 bit 1
        if (pf != pf1_data[1]) begin
          $display("pf1 bit 1 error! pf: %d, pf1_data[1]: %d", pf, pf1_data[1]);
          $finish;
        end
      end else if (cc >= 148 + 44 && cc < 148 + 48) begin // pf1 bit 0
        if (pf != pf1_data[0]) begin
          $display("pf1 bit 0 error! pf: %d, pf1_data[0]: %d", pf, pf1_data[0]);
          $finish;
        end
      end else if (cc >= 148 + 48 && cc < 148 + 52) begin // pf2 bit 0
        if (pf != pf2_data[0]) begin
          $display("pf2 bit 0 error! pf: %d, pf2_data[0]: %d", pf, pf2_data[0]);
          $finish;
        end
      end else if (cc >= 148 + 52 && cc < 148 + 56) begin // pf2 bit 1
        if (pf != pf2_data[1]) begin
          $display("pf2 bit 1 error! pf: %d, pf2_data[1]: %d", pf, pf2_data[1]);
          $finish;
        end
      end else if (cc >= 148 + 56 && cc < 148 + 60) begin // pf2 bit 2
        if (pf != pf2_data[2]) begin
          $display("pf2 bit 2 error! pf: %d, pf2_data[2]: %d", pf, pf2_data[2]);
          $finish;
        end
      end else if (cc >= 148 + 60 && cc < 148 + 64) begin // pf2 bit 3
        if (pf != pf2_data[3]) begin
          $display("pf2 bit 3 error! pf: %d, pf2_data[3]: %d", pf, pf2_data[3]);
          $finish;
        end
      end else if (cc >= 148 + 64 && cc < 148 + 68) begin // pf2 bit 4
        if (pf != pf2_data[4]) begin
          $display("pf2 bit 4 error! pf: %d, pf2_data[4]: %d", pf, pf2_data[4]);
          $finish;
        end
      end else if (cc >= 148 + 68 && cc < 148 + 72) begin // pf2 bit 5
        if (pf != pf2_data[5]) begin
          $display("pf2 bit 5 error! pf: %d, pf2_data[5]: %d", pf, pf2_data[5]);
          $finish;
        end
      end else if (cc >= 148 + 72 && cc < 148 + 76) begin // pf2 bit 6
        if (pf != pf2_data[6]) begin
          $display("pf2 bit 6 error! pf: %d, pf2_data[6]: %d", pf, pf2_data[6]);
          $finish;
        end
      end else begin                                      // pf2 bit 7
        if (pf != pf2_data[7]) begin
          $display("pf2 bit 7 error! pf: %d, pf2_data[7]: %d", pf, pf2_data[7]);
          $finish;
        end
      end
    end else begin
      if (cc >= 148 +  0 && cc < 148 +  4) begin          // pf2 bit 7
        if (pf != pf2_data[7]) begin
          $display("pf2 bit 7 error! pf: %d, pf2_data[7]: %d", pf, pf2_data[7]);
          $finish;
        end
      end else if (cc >= 148 +  4 && cc < 148 +  8) begin // pf2 bit 6
        if (pf != pf2_data[6]) begin
          $display("pf2 bit 6 error! pf: %d, pf2_data[6]: %d", pf, pf2_data[6]);
          $finish;
        end
      end else if (cc >= 148 +  8 && cc < 148 + 12) begin // pf2 bit 5
        if (pf != pf2_data[5]) begin
          $display("pf2 bit 5 error! pf: %d, pf2_data[5]: %d", pf, pf2_data[5]);
          $finish;
        end
      end else if (cc >= 148 + 12 && cc < 148 + 16) begin // pf2 bit 4
        if (pf != pf2_data[4]) begin
          $display("pf2 bit 4 error! pf: %d, pf2_data[4]: %d", pf, pf2_data[4]);
          $finish;
        end
      end else if (cc >= 148 + 16 && cc < 148 + 20) begin // pf2 bit 3
        if (pf != pf2_data[3]) begin
          $display("pf2 bit 3 error! pf: %d, pf2_data[3]: %d", pf, pf2_data[3]);
          $finish;
        end
      end else if (cc >= 148 + 20 && cc < 148 + 24) begin // pf2 bit 2
        if (pf != pf2_data[2]) begin
          $display("pf2 bit 2 error! pf: %d, pf2_data[2]: %d", pf, pf2_data[2]);
          $finish;
        end
      end else if (cc >= 148 + 24 && cc < 148 + 28) begin // pf2 bit 1
        if (pf != pf2_data[1]) begin
          $display("pf2 bit 1 error! pf: %d, pf2_data[1]: %d", pf, pf2_data[1]);
          $finish;
        end
      end else if (cc >= 148 + 28 && cc < 148 + 32) begin // pf2 bit 0
        if (pf != pf2_data[0]) begin
          $display("pf2 bit 0 error! pf: %d, pf2_data[0]: %d", pf, pf2_data[0]);
          $finish;
        end
      end else if (cc >= 148 + 32 && cc < 148 + 36) begin // pf1 bit 0
        if (pf != pf1_data[0]) begin
          $display("pf1 bit 0 error! pf: %d, pf1_data[0]: %d", pf, pf1_data[0]);
          $finish;
        end
      end else if (cc >= 148 + 36 && cc < 148 + 40) begin // pf1 bit 1
        if (pf != pf1_data[1]) begin
          $display("pf1 bit 1 error! pf: %d, pf1_data[1]: %d", pf, pf1_data[1]);
          $finish;
        end
      end else if (cc >= 148 + 40 && cc < 148 + 44) begin // pf1 bit 2
        if (pf != pf1_data[2]) begin
          $display("pf1 bit 2 error! pf: %d, pf1_data[2]: %d", pf, pf1_data[2]);
          $finish;
        end
      end else if (cc >= 148 + 44 && cc < 148 + 48) begin // pf1 bit 3
        if (pf != pf1_data[3]) begin
          $display("pf1 bit 3 error! pf: %d, pf1_data[3]: %d", pf, pf1_data[3]);
          $finish;
        end
      end else if (cc >= 148 + 48 && cc < 148 + 52) begin // pf1 bit 4
        if (pf != pf1_data[4]) begin
          $display("pf1 bit 4 error! pf: %d, pf1_data[4]: %d", pf, pf1_data[4]);
          $finish;
        end
      end else if (cc >= 148 + 52 && cc < 148 + 56) begin // pf1 bit 5
        if (pf != pf1_data[5]) begin
          $display("pf1 bit 5 error! pf: %d, pf1_data[5]: %d", pf, pf1_data[5]);
          $finish;
        end
      end else if (cc >= 148 + 56 && cc < 148 + 60) begin // pf1 bit 6
        if (pf != pf1_data[6]) begin
          $display("pf1 bit 6 error! pf: %d, pf1_data[6]: %d", pf, pf1_data[6]);
          $finish;
        end
      end else if (cc >= 148 + 60 && cc < 148 + 64) begin // pf1 bit 7
        if (pf != pf1_data[7]) begin
          $display("pf1 bit 7 error! pf: %d, pf1_data[7]: %d", pf, pf1_data[7]);
          $finish;
        end
      end else if (cc >= 148 + 64 && cc < 148 + 68) begin // pf0 bit 7
        if (pf != pf0_data[3]) begin
          $display("pf0 bit 7 error! pf: %d, pf0_data[3]: %d", pf, pf0_data[3]);
          $finish;
        end
      end else if (cc >= 148 + 68 && cc < 148 + 72) begin // pf0 bit 6
        if (pf != pf0_data[2]) begin
          $display("pf0 bit 6 error! pf: %d, pf0_data[2]: %d", pf, pf0_data[2]);
          $finish;
        end
      end else if (cc >= 148 + 72 && cc < 148 + 76) begin // pf0 bit 5
        if (pf != pf0_data[1]) begin
          $display("pf0 bit 5 error! pf: %d, pf0_data[1]: %d", pf, pf0_data[1]);
          $finish;
        end
      end else begin                                      // pf0 bit 4
        if (pf != pf0_data[0]) begin
          $display("pf0 bit 4 error! pf: %d, pf0_data[0]: %d", pf, pf0_data[0]);
          $finish;
        end
      end
    end
  end else begin
    line_count = line_count + 1;
    cc = 0;
    case (line_count)
      1: begin
        pf0_data = 4'b0101;
        pf1_data = 8'b01010101;
        pf2_data = 8'b01010101;
      end
      2: begin
        ref_bar = 1;
      end
      3: begin
        pf0_data = 4'b1010;
        pf1_data = 8'b10101010;
        pf2_data = 8'b10101010;
      end
      4: begin
        ref_bar = 0;
      end
      5: begin
        $display("OK");
        $finish;
      end
    endcase
  end
end

endmodule  // tia_playfield_registers_sim

