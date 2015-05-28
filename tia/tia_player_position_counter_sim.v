`include "tia_player_position_counter.v"

module tia_player_position_counter_sim();

reg motck, p0ec_bar, p0re, nz0, nz1, nz2;
wire start_bar, fstob, pck, pphi1, pphi2, count_bar;

wire nz0_bar, nz1_bar, nz2_bar;
assign nz0_bar = ~nz0;
assign nz1_bar = ~nz1;
assign nz2_bar = ~nz2;

integer cc;
integer line_count;

tia_player_position_counter c(.motck(motck),
                              .pec_bar(p0ec_bar),
                              .pre(p0re),
                              .nz0_bar(nz0_bar),
                              .nz1_bar(nz1_bar),
                              .nz2_bar(nz2_bar),
                              .start_bar(start_bar),
                              .fstob(fstob),
                              .pck(pck),
                              .pphi1(pphi1),
                              .pphi2(pphi2),
                              .count_bar(count_bar));

initial begin
  motck = 0;
  p0ec_bar = 1;
  p0re = 1;
  nz0 = 0;
  nz1 = 0;
  nz2 = 0;
  cc = 0;
  line_count = 0;

  $dumpfile("out/tia_player_position_counter_sim.vcd");
  $dumpvars(0, tia_player_position_counter_sim);
end

always #100 begin
  motck = ~motck;
end

always @(posedge motck) begin
  #2
  if (cc < 4) begin
    if (cc == 1 && line_count == 0) begin
      p0re = 0;
    end
    if (start_bar != 0) begin
      $display("missing first start_bar at cc: %d, line: %d", cc, line_count);
      $finish;
    end
  end else if ((line_count == 1 || line_count == 3) &&
               (cc >= 16 && cc < 20)) begin
    if (start_bar != 0) begin
      $display("missing close start_bar at cc: %d, line: %d", cc, line_count);
      $finish;
    end
  end else if ((line_count == 2 || line_count == 3 || line_count == 6) &&
               (cc >= 32 && cc < 36)) begin
    if (start_bar != 0) begin
      $display("missing medium start_bar at cc: %d, line: %d", cc, line_count);
      $finish;
    end
  end else if ((line_count == 4 || line_count == 6) &&
               (cc >= 64 && cc < 68)) begin
    if (start_bar != 0) begin
      $display("missing far start_bar at cc: %d, line: %d", cc, line_count);
      $finish;
    end
  end else begin
    if (start_bar != 1) begin
      $display("extraneous start_bar at cc: %d, line: %d", cc, line_count);
      $finish;
    end
  end
  cc = cc + 1;
  if (cc > 159) begin
    cc = 0;
    line_count = line_count + 1;
    case (line_count)
      1: begin
        nz0 = 1;
        nz1 = 0;
        nz2 = 0;
      end
      2: begin
        nz0 = 0;
        nz1 = 1;
        nz2 = 0;
      end
      3: begin
        nz0 = 1;
        nz1 = 1;
        nz2 = 0;
      end
      4: begin
        nz0 = 0;
        nz1 = 0;
        nz2 = 1;
      end
      5: begin
        nz0 = 1;
        nz1 = 0;
        nz2 = 1;
      end
      6: begin
        nz0 = 0;
        nz1 = 1;
        nz2 = 1;
      end
      7: begin
        nz0 = 1;
        nz1 = 1;
        nz2 = 1;
      end
      8: begin
        $display("OK");
        $finish;
      end
    endcase
  end
end

endmodule  // tia_player_position_counter_sim
