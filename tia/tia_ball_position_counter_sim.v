`include "tia_ball_position_counter.v"

module tia_ball_position_counter_sim();

reg motck, blec_bar, blre, blen, p1gr, blvd, blsiz2, blsiz1, clkp, d1, d0;
wire bl;

integer cc;
integer line_count;

wire blsiz1_bar;
assign blsiz1_bar = ~blsiz1;

tia_ball_position_counter ball(.motck(motck),
                               .blec_bar(blec_bar),
                               .blre(blre),
                               .blen(blen),
                               .p1gr(p1gr),
                               .blvd(blvd),
                               .blsiz2(blsiz2),
                               .blsiz1_bar(blsiz1_bar),
                               .clkp(clkp),
                               .d1(d1),
                               .d0(d0),
                               .bl(bl));

initial begin
  motck = 0;
  blec_bar = 1;
  blre = 1;
  blen = 1;
  p1gr = 0;
  blvd = 1;  // latch the 0 in d0, disabling ball vertical delay
  blsiz2 = 0;
  blsiz1 = 0;
  clkp = 0;
  d1 = 1;
  d0 = 0;

  cc = 0;
  line_count = 0;
end

always #100 begin
  motck = ~motck;
  clkp = ~clkp;
end

always @(posedge motck) begin
  if (line_count == 0 && cc == 0) begin
    blre = 0;
    blen = 1;
  end

  if ((line_count == 1 && cc < 1) ||
      (line_count == 2 && cc < 2) ||
      (line_count == 3 && cc < 4) ||
      (line_count == 4 && cc < 8)) begin
    if (bl != 1) begin
      $display("missing ball pixel %d on line %d", cc, line_count);
      $finish;
    end
  end else if (line_count > 0) begin
    if (bl != 0) begin
      $display("spurious ball graphics, cc: %d, line: %d", cc, line_count);
      $finish;
    end
  end

  cc = cc + 1;
  if (cc == 160) begin
    cc = 0;
    line_count = line_count + 1;
    case (line_count)
      2: begin
        blsiz1 = 1;
        blsiz2 = 0;
      end
      3: begin
        blsiz1 = 0;
        blsiz2 = 1;
      end
      4: begin
        blsiz1 = 1;
        blsiz2 = 1;
      end
      5: begin
        $display("OK");
        $finish;
      end
    endcase
  end
end

endmodule  // tia_ball_position_counter_sim
