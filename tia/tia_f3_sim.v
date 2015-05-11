`include "tia_f3.v"

module tia_f3_sim();

reg clock;
reg reset;
reg s1, s2;
reg r1, r2;
wire q;
wire q_bar;

integer cycle_count;

tia_f3 f3(.s1(s1),
          .s2(s2),
          .r1(r1),
          .r2(r2),
          .clock(clock),
          .reset(reset),
          .q(q),
          .q_bar(q_bar));

initial begin
  clock = 1;
  reset = 1;
  s1 = 0;
  s2 = 0;
  r1 = 1;
  r2 = 0;
  cycle_count = 0;
end

always #100 begin
  clock = ~clock;
end

always @(negedge clock) begin
  case (cycle_count)
    0: begin  // test set
      reset = 0;
      s1 = 0;
      r1 = 1;
    end
    1: begin  // test holding set value
      reset = 0;
      s1 = 1;
      r1 = 1;
    end
    2: begin  // test reset
      reset = 0;
      s1 = 1;
      r1 = 0;
    end
    3: begin  // test hold reset
      reset = 0;
      s1 = 1;
      r1 = 1;
    end

    4: begin  // repeat tests with reset true, should clobber output
      reset = 1;
      s1 = 0;
      r1 = 1;
      if (!(!q && q_bar)) begin
        $display("- reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    5: begin  // test holding set value
      reset = 1;
      s1 = 1;
      r1 = 1;
      if (!(!q && q_bar)) begin
        $display("- reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    6: begin  // test reset
      reset = 1;
      s1 = 1;
      r1 = 0;
      if (!(!q && q_bar)) begin
        $display("- reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    7: begin  // test hold reset
      reset = 1;
      s1 = 1;
      r1 = 1;
      if (!(!q && q_bar)) begin
        $display("- reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
  endcase
end

always @(posedge clock) begin
  #5
  case (cycle_count)
    0: begin  // test set
      if (q != 1 || q_bar != 0) begin
        $display("s FAIL, q = %h, q_bar = %h", q, q_bar);
        $finish;
      end
    end
    1: begin  // test hold set
      if (q != 1 && q_bar != 0) begin
        $display("s hold FAIL, q = %h, q_bar = %h", q, q_bar);
        $finish;
      end
    end
    2: begin  // test reset
      if (q != 0 || q_bar != 1) begin
        $display("r FAIL, q = %h, q_bar = %h", q, q_bar);
        $finish;
      end
    end
    3: begin  // test hold reset
      if (q != 0 || q_bar != 1) begin
        $display("r hold FAIL, q = %h, q_bar = %h", q, q_bar);
        $finish;
      end
    end

    4: begin  // test set
      if (q != 0 || q_bar != 1) begin
        $display("+ reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    5: begin  // test hold set
      if (q != 0 || q_bar != 1) begin
        $display("+ reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    6: begin  // test reset
      if (q != 0 || q_bar != 1) begin
        $display("+ reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    7: begin  // test hold reset
      if (q != 0 || q_bar != 1) begin
        $display("+ reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    8: begin
      $display("OK");
      $finish;
    end
  endcase
  cycle_count = cycle_count + 1;
end

endmodule  // tia_f3_sim
