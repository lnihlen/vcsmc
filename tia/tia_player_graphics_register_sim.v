`include "tia_player_graphics_register.v"

module tia_player_graphics_register_sim();

reg[7:0] d;
reg[2:0] gs;
reg p0gr, p1gr;

wire new, old;
wire gs0, gs1, gs2;
assign gs0 = gs[0];
assign gs1 = gs[1];
assign gs2 = gs[2];

wire d0, d1, d2, d3, d4, d5, d6, d7;
assign d0 = d[0];
assign d1 = d[1];
assign d2 = d[2];
assign d3 = d[3];
assign d4 = d[4];
assign d5 = d[5];
assign d6 = d[6];
assign d7 = d[7];

reg up;

tia_player_graphics_register pgr(.d0(d0), .d1(d1), .d2(d2), .d3(d3), .d4(d4),
    .d5(d5), .d6(d6), .d7(d7), .gs0(gs0), .gs1(gs1), .gs2(gs2), .p0gr(p0gr),
    .p1gr(p1gr), .new(new), .old(old));

initial begin
  d = 8'b0000000;
  gs = 3'b000;
  p0gr = 1;
  p1gr = 0;

  up = 1;
end

always #100 begin
  p0gr = 0;

  // TODO: old testing as well
  case (gs)
    0: begin
      if (~new != d0) begin
        $display("error d: %b, gs: %b, new: %d, old: %d", d, gs, new, old);
        $finish;
      end
    end
    1: begin
      if (~new != d1) begin
        $display("error d: %b, gs: %b, new: %d, old: %d", d, gs, new, old);
        $finish;
      end
    end
    2: begin
      if (~new != d2) begin
        $display("error d: %b, gs: %b, new: %d, old: %d", d, gs, new, old);
        $finish;
      end
    end
    3: begin
      if (~new != d3) begin
        $display("error d: %b, gs: %b, new: %d, old: %d", d, gs, new, old);
        $finish;
      end
    end
    4: begin
      if (~new != d4) begin
        $display("error d: %b, gs: %b, new: %d, old: %d", d, gs, new, old);
        $finish;
      end
    end
    5: begin
      if (~new != d5) begin
        $display("error d: %b, gs: %b, new: %d, old: %d", d, gs, new, old);
        $finish;
      end
    end
    6: begin
      if (~new != d6) begin
        $display("error d: %b, gs: %b, new: %d, old: %d", d, gs, new, old);
        $finish;
      end
    end
    7: begin
      if (~new != d7) begin
        $display("error d: %b, gs: %b, new: %d, old: %d", d, gs, new, old);
        $finish;
      end
    end
  endcase

  if (!up && gs == 0) begin
    if (d == 8'b11111111) begin
      $display("OK");
      $finish;
    end
    d = d + 1;
    p0gr = 1;
  end

  if (up) begin
    gs = gs + 1;
    if (gs == 0) begin
      gs = 7;
      up = 0;
    end
  end else begin
    gs = gs + 7;
    if (gs == 7) begin
      gs = 0;
      up = 1;
    end
  end
end

endmodule  // tia_player_graphics_register_sim
