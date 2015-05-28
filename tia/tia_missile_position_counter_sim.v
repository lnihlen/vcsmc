`include "tia_missile_position_counter.v"

module tia_missile_position_counter_sim();

reg motck, mec_bar, clkp, nz0, nz1, nz2, missile_enable,
    missile_to_player_reset_bar, nz4, nz5, missile_reset,
    missile_to_player_reset;

wire m;
wire nz0_bar, nz1_bar, nz2_bar, nz4_bar, nz5_bar;
assign nz0_bar = ~nz0;
assign nz1_bar = ~nz1;
assign nz2_bar = ~nz2;
assign nz4_bar = ~nz4;
assign nz5_bar = ~nz5;
integer cc;
integer line_count;

tia_missile_position_counter mpc(
    .motck(motck),
    .mec_bar(mec_bar),
    .clkp(clkp),
    .nz0_bar(nz0_bar),
    .nz1_bar(nz1_bar),
    .nz2_bar(nz2_bar),
    .missile_enable(missile_enable),
    .missile_to_player_reset_bar(missile_to_player_reset_bar),
    .nz4_bar(nz4_bar),
    .nz5_bar(nz5_bar),
    .missile_reset(missile_reset),
    .missile_to_player_reset(missile_to_player_reset),
    .m(m));

initial begin
  motck = 0;
  mec_bar = 1;
  clkp = 0;
  nz0 = 0;
  nz1 = 0;
  nz2 = 0;
  missile_enable = 1;
  missile_to_player_reset_bar = 1;
  nz4 = 0;
  nz5 = 0;
  missile_reset = 0;
  missile_to_player_reset = 0;
  cc = 0;
  line_count = 0;

  $dumpfile("out/tia_missile_position_counter_sim.vcd");
  $dumpvars(0, tia_missile_position_counter_sim);
end

always #100 begin
  motck = ~motck;
  clkp = ~clkp;
end

always @(posedge motck) begin
  #1
  // TODO: test nz4, nz5
  if (cc == 0) begin
    if (m != 1) begin
      $display("missing startup missile line: %d", line_count);
      $finish;
    end
  end else if ((line_count == 2 || line_count == 4) && cc == 16) begin
    if (m != 1) begin
      $display("missing close missile at 16 line: %d", line_count);
      $finish;
    end
  end else if ((line_count == 3 || line_count == 4 || line_count == 7) &&
               cc == 32) begin
    if (m != 1) begin
      $display("missing medium missile at 32 line: %d", line_count);
      $finish;
    end
  end else if ((line_count == 5 || line_count == 7) && cc == 64) begin
    if (m != 1) begin
      $display("missing far missile at 64 line: %d", line_count);
      $finish;
    end
  end else if (m != 0) begin
    $display("extra missile activation, line: %d, cc: %d", line_count, cc);
  end

  cc = cc + 1;
  if (cc == 160) begin
    line_count = line_count + 1;
    cc = 0;
    // TODO: this is hiding the fact that the first scan line goes by without
    // the missile firing at 0. Ensure that is the desired behavior.
    case (line_count)
      1: begin
        nz0 = 0;
        nz1 = 0;
        nz2 = 0;
      end
      2: begin
        nz0 = 1;
        nz1 = 0;
        nz2 = 0;
      end
      3: begin
        nz0 = 0;
        nz1 = 1;
        nz2 = 0;
      end
      4: begin
        nz0 = 1;
        nz1 = 1;
        nz2 = 0;
      end
      5: begin
        nz0 = 0;
        nz1 = 0;
        nz2 = 1;
      end
      6: begin
        nz0 = 1;
        nz1 = 0;
        nz2 = 1;
      end
      7: begin
        nz0 = 0;
        nz1 = 1;
        nz2 = 1;
      end
      8: begin
        nz0 = 1;
        nz1 = 1;
        nz2 = 1;
      end
      9: begin
        $display("OK");
        $finish;
      end
    endcase
  end

end

endmodule  // tia_missile_position_counter_sim
