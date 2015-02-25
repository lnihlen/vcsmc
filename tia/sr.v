// SR latch, used throughout the TIA.
module sr(s, r, q, q_bar);

input s, r;
output q, q_bar;

wire s;
wire r;
reg q;
wire q_bar;

assign q_bar = ~q;

initial begin
  q = 0;
end

always @(s, r) begin
  if (s === 0 && r === 1)
    q = 1;
  else if (s === 1 && r === 0)
    q = 0;
  else if (s === 1 && r === 1) begin
    $display("ERROR s and r both 1 in SR flip-flip.");
    $stop;
  end
end

endmodule  // sr
