`include "tia_biphase_clock.v"
`include "tia_dl.v"

module tia_dl_sim();

reg dl_in;
reg dl_r;
wire s1;
wire s2;
reg clock;
wire dl_out;
reg rsyn;
wire rsynl;
wire tap;

integer cycle_count;

tia_biphase_clock bpc(.clk(clock),
                      .r(rsyn),
                      .phi1(s1),
                      .phi2(s2),
                      .rl(rsynl));

tia_dl dl(.in(dl_in),
          .s1(s1),
          .s2(s2),
          .r(dl_r),
          .out(dl_out));

initial begin
  dl_in = 0;
  dl_r = 0;
  clock = 0;
  rsyn = 0;
  cycle_count = 0;
end

always #100 begin
  clock = ~clock;
end

always @(posedge s1) begin
  case (cycle_count)
    1: begin
      dl_in = 1;
    end
    3: begin
      dl_in = 0;
    end
    5: begin
      dl_r = 1;
    end
    7: begin
      dl_in = 1;
    end
    9: begin
      dl_r = 0;
    end
    11: begin
      dl_in = 0;
    end
  endcase
end

always @(posedge s2) begin
  #1
  if (cycle_count == 0) begin
    if (dl_out != 0) begin
      $display("error cycle: %d, in: %d, out: %d, r: %d",
          cycle_count, dl_in, dl_out, dl_r);
      $finish;
    end
  end else if (cycle_count < 5) begin
    if (dl_out != 1) begin
      $display("error cycle: %d, in: %d, out: %d, r: %d",
          cycle_count, dl_in, dl_out, dl_r);
      $finish;
    end
  end else if (cycle_count < 9) begin
    if (dl_out != 0) begin
      $display("error cycle: %d, in: %d, out: %d, r: %d",
          cycle_count, dl_in, dl_out, dl_r);
      $finish;
    end
  end else if (cycle_count < 11) begin
    if (dl_out != 1) begin
      $display("error cycle: %d, in: %d, out: %d, r: %d",
          cycle_count, dl_in, dl_out, dl_r);
      $finish;
    end
  end else begin
    $display("OK");
    $finish;
  end
  cycle_count = cycle_count + 1;
end

endmodule  // tia_dl_sim
