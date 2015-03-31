`include "tia_f1.v"

module tia_f1_sim();

reg clock;
reg reset;
reg s;
reg r;
wire q;
wire q_bar;

integer cycle_count;

tia_f1 f1(.s(s),
          .r(r),
          .clock(clock),
          .reset(reset),
          .q(q),
          .q_bar(q_bar));

initial begin
  clock = 1;
  reset = 1;
  s = 0;
  r = 1;
  cycle_count = 0;
end

always #100 begin
  clock = ~clock;
end

always @(negedge clock) begin
  case (cycle_count)
    0: begin  // test set
      reset = 0;
      s = 0;
      r = 1;
    end
    1: begin  // test holding set value
      reset = 0;
      s = 1;
      r = 1;
    end
    2: begin  // test reset
      reset = 0;
      s = 1;
      r = 0;
    end
    3: begin  // test hold reset
      reset = 0;
      s = 1;
      r = 1;
    end

    4: begin  // repeat tests with reset true, should clobber output
      reset = 1;
      s = 0;
      r = 1;
      if (!(!q && q_bar)) begin
        $display("- reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    5: begin  // test holding set value
      reset = 1;
      s = 1;
      r = 1;
      if (!(!q && q_bar)) begin
        $display("- reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    6: begin  // test reset
      reset = 1;
      s = 1;
      r = 0;
      if (!(!q && q_bar)) begin
        $display("- reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
    7: begin  // test hold reset
      reset = 1;
      s = 1;
      r = 1;
      if (!(!q && q_bar)) begin
        $display("- reset FAIL, q: %h, q_bar: %h", q, q_bar);
        $finish;
      end
    end
  endcase
end

always @(posedge clock) begin
  #10
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

endmodule  // tia_f1_sim
