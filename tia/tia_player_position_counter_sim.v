`include "tia_player_position_counter.v"

module tia_player_position_counter_sim();

reg motck, p0ec_bar, p0re, nz0, nz1, nz2;
wire start_bar, fstob, pck, pphi1, pphi2, count_bar;

integer cc;

tia_player_position_counter c(.motck(motck),
                              .p0ec_bar(p0ec_bar),
                              .p0re(p0re),
                              .nz0(nz0),
                              .nz1(nz1),
                              .nz2(nz2),
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
  nz0 = 1;
  nz1 = 1;
  nz2 = 1;
  cc = 0;
end

always #100 begin
  motck = ~motck;
  #1
  if (motck) cc = cc + 1;
end

always @(posedge motck) begin
  #3
  if (cc >= 0 && cc < 4) begin
    if (cc == 2) begin
      p0re = 0;
    end
    if (start_bar != 0) begin
      $display("missing first start_bar at cc: %d", cc);
      $finish;
    end
  end else if (cc >= 16 && cc < 20) begin
    if (start_bar != 0) begin
      $display("missing close start_bar at cc: %d", cc);
      $finish;
    end
  end else if (cc >= 32 && cc < 36) begin
    if (start_bar != 0) begin
      $display("missing medium start_bar at cc: %d", cc);
      $finish;
    end
  end else if (cc >= 64 && cc < 68) begin
    if (start_bar != 0) begin
      $display("missing far start_bar at cc: %d", cc);
      $finish;
    end
  end else if (cc >= 160 && cc < 164) begin
    if (start_bar != 0) begin
      $display("missing restart start_bar at cc: %d", cc);
      $finish;
    end
  end else if (cc > 165) begin
    $display("OK");
    $finish;
  end else begin
    if (start_bar != 1) begin
      $display("extraneous start_bar at cc: %d", cc);
      $finish;
    end
  end
end

endmodule  // tia_player_position_counter_sim
