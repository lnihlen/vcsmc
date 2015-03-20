`include "sr.v"

module sr_sim();

reg s;
reg r;
wire q;
wire q_bar;
reg clock;

integer cycle_count;

sr sr(.s(s), .r(r), .r2(0), .q(q), .q_bar(q_bar));

initial begin
  s = 0;  // set
  r = 1;
  cycle_count = 0;
  clock = 0;
end

always #10 clock = ~clock;

always @(posedge clock) begin
  case (cycle_count)
    0: begin  // set
      s = 0;
      r = 1;
    end
    1: begin  // reset
      s = 1;
      r = 0;
    end
    3: begin  // set
      s = 0;
      r = 1;
    end
    default: begin
      s = 0;
      r = 0;
    end
  endcase
end

always @(negedge clock) begin
  case (cycle_count)
    0: begin  // set
      if (q != 1 || q_bar != 0) begin
        $display("ERROR s = %d, r = %d, q = %d, q_bar = %d", s, r, q, q_bar);
        $finish;
      end
    end
    1: begin  // reset
      if (q != 0 || q_bar != 1) begin
        $display("ERROR s = %d, r = %d, q = %d, q_bar = %d", s, r, q, q_bar);
        $finish;
      end
    end
    2: begin  // hold reset
      if (q != 0 || q_bar != 1) begin
        $display("ERROR s = %d, r = %d, q = %d, q_bar = %d", s, r, q, q_bar);
        $finish;
      end
    end
    3: begin  // set
      if (q != 1 || q_bar != 0) begin
        $display("ERROR s = %d, r = %d, q = %d, q_bar = %d", s, r, q, q_bar);
        $finish;
      end
    end
    4: begin  // hold set
      if (q != 1 || q_bar != 0) begin
        $display("ERROR s = %d, r = %d, q = %d, q_bar = %d", s, r, q, q_bar);
        $finish;
      end
    end
    5: begin
      $display("OK");
      $finish;
    end
  endcase
  cycle_count = cycle_count + 1;
end

endmodule  // sr_sim
