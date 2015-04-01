`include "tia_player_graphics_scan_counter.v"

module tia_player_graphics_scan_counter_sim();

reg start_bar, fstob, pck, count_bar, new, old, player_vert_delay_bar;
reg missile_to_player_reset_bar, player_reflect_bar, clkp;
wire missile_to_player_reset, gs0, gs1, gs2, p;
integer cc;

tia_player_graphics_scan_counter pgsc(
    .start_bar(start_bar),
    .fstob(fstob),
    .pck(pck),
    .count_bar(count_bar),
    .new(new),
    .old(old),
    .player_vert_delay_bar(player_vert_delay_bar),
    .missile_to_player_reset_bar(missile_to_player_reset_bar),
    .player_reflect_bar(player_reflect_bar),
    .clkp(clkp),
    .missile_to_player_reset(missile_to_player_reset),
    .gs0(gs0),
    .gs1(gs1),
    .gs2(gs2),
    .p(p));

initial begin
  start_bar = 1;
  fstob = 0;
  pck = 0;
  count_bar = 0;
  new = 0;
  old = 0;
  player_vert_delay_bar = 1;
  missile_to_player_reset_bar = 1;
  player_reflect_bar = 0;
  clkp = 0;
  cc = 0;
end

always #100 begin
  clkp = ~clkp;
  pck = ~pck;
end


always @(posedge pck) begin
  #1
  case (cc)
    2: start_bar = 0;
    3: start_bar = 1;
    // TODO: note unusual counting order, double-back to verify this is sane
    4: begin
      if (gs2 != 0 || gs1 != 0 || gs0 != 1) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    5: begin
      if (gs2 != 0 || gs1 != 1 || gs0 != 0) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    6: begin
      if (gs2 != 0 || gs1 != 1 || gs0 != 1) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    7: begin
      if (gs2 != 1 || gs1 != 0 || gs0 != 0) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    8: begin
      if (gs2 != 1 || gs1 != 0 || gs0 != 1) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    9: begin
      if (gs2 != 1 || gs1 != 1 || gs0 != 0) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    10: begin
      if (gs2 != 1 || gs1 != 1 || gs0 != 1) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    11: begin
      if (gs2 != 0 || gs1 != 0 || gs0 != 0) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    12: player_reflect_bar = 1;
    13: start_bar = 0;
    14: start_bar = 1;
    15: begin
      if (gs2 != 1 || gs1 != 1 || gs0 != 0) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    16: begin
      if (gs2 != 1 || gs1 != 0 || gs0 != 1) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    17: begin
      if (gs2 != 1 || gs1 != 0 || gs0 != 0) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    18: begin
      if (gs2 != 0 || gs1 != 1 || gs0 != 1) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    19: begin
      if (gs2 != 0 || gs1 != 1 || gs0 != 0) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    20: begin
      if (gs2 != 0 || gs1 != 0 || gs0 != 1) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    21: begin
      if (gs2 != 0 || gs1 != 0 || gs0 != 0) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    22: begin
      if (gs2 != 1 || gs1 != 1 || gs0 != 1) begin
        $display("error cycle %d", cc);
        $finish;
      end
    end
    23: begin
      $display("OK");
      $finish;
    end
  endcase

  if ((cc >= 4 && cc < 12) || (cc >= 15 && cc < 23)) begin
    if (p != 1) begin
      $display("missing player activation: %d", cc);
      $finish;
    end
  end else begin
    if (p != 0) begin
      $display("extraneous player activation: %d", cc);
      $finish;
    end
  end
  #1
  cc = cc + 1;
end

endmodule  // tia_player_graphics_scan_counter_sim
