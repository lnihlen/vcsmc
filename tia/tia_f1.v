// F1 block defined on TIA schematics page 1, section C-1.
module tia_f1(s, r, clock, reset, q, q_bar);

input s;
input r;
input clock;
input reset;

output q;
output q_bar;

wire s;
wire r;
wire clock;
wire reset;
reg mid_s;  // storage for first SR flip-flop
reg mid_r;
reg q;
wire q_bar;

assign q_bar = ~q;

initial begin
  mid_s = 1;
  mid_r = 0;
  q = 0;
end

always @(posedge reset) begin
  mid_s = 1;
  mid_r = 0;
  q = 0;
end

// First NOR gate SR flip-flop will has CK attached to NORs on input with s
// and r, meaning that ck being 1 will force s and r to 0, rendering the
// first gate inactive when CK is 1.
always @(negedge clock) begin
  if (reset === 1) begin
    mid_s = 1;
    mid_r = 0;
    q = 0;
  end else if (s === 1 && r === 1) begin
    mid_s = 0;
    mid_r = 0;
  end else if (s === 1 && r === 0) begin
    mid_s = 1;
    mid_r = 0;
  end else if (s === 0 && r === 1) begin
    mid_s = 0;
    mid_r = 1;
  end else if (s === 0 && r === 0) begin
    $display("ERROR at time %d: s and r both 0 in F1.", $time);
    $stop;
  end
end

// Second NOR gate SR flip flop is AND gate with clock on inputs.
always @(posedge clock) begin
  if (reset === 1) begin
    mid_s = 1;
    mid_r = 0;
    q = 0;
  end else if (mid_s === 0 && mid_r === 1) begin
    q = 1;
  end else if (mid_s === 1 && mid_r === 0) begin
    q = 0;
  end
end

endmodule  // tia_f1
