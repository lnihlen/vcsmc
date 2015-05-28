`include "tia_l.v"

module tia_l_sim();

reg f;
reg l;
reg in;
reg clock;
wire out;
wire out_bar;

integer cycle_count;

tia_l test_latch(.in(in), .follow(f), .latch(l), .out(out), .out_bar(out_bar));

initial begin
  f = 1;
  l = 0;
  in = 0;
  clock = 0;
  cycle_count = 0;

  $dumpfile("out/tia_l_sim.vcd");
  $dumpvars(0, tia_l_sim);
end

always #100 begin
  clock = ~clock;
end

always @(posedge clock) begin
  case (cycle_count)
    1: begin
      #2
      if (out != 0 || out_bar != 1) begin
        $display("ERROR cycle_count: %d", cycle_count);
        $finish;
      end
    end
    2: begin
      in = 1;
      #2
      if (out != 1 || out_bar != 0) begin
        $display("ERROR cycle_count: %d", cycle_count);
        $finish;
      end
    end
    3: begin
      f = 0;
      l = 1;
      #2
      if (out != 1 || out_bar != 0) begin
        $display("ERROR cycle_count: %d", cycle_count);
        $finish;
      end
    end
    4: begin
      in = 0;
      #2
      if (out != 1 || out_bar != 0) begin
        $display("ERROR cycle_count: %d", cycle_count);
        $finish;
      end
    end
    5: begin
      f = 1;
      l = 0;
      #2
      if (out != 0 || out_bar != 1) begin
        $display("ERROR cycle_count: %d", cycle_count);
        $finish;
      end
    end
    6: begin
      f = 0;
      l = 1;
      #2
      if (out != 0 || out_bar != 1) begin
        $display("ERROR cycle_count: %d", cycle_count);
        $finish;
      end
    end
    6: begin
      in = 1;
      #2
      if (out != 0 || out_bar != 1) begin
        $display("ERROR cycle_count: %d", cycle_count);
        $finish;
      end
    end
    7: begin
      $display("OK");
      $finish;
    end
  endcase
  cycle_count = cycle_count + 1;
end

endmodule  // tia_l_sim
