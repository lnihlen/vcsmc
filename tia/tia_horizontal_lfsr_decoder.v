module tia_horizontal_lfsr_decoder(in, rhs, cnt, rcb, shs, lrhb, rhb);

input[5:0] in;
output rhs, cnt, rcb, shs, lrhb, rhb;

wire[5:0] in;
wire rhs;
wire cnt;
wire rcb;
wire shs;
wire lrhb;
wire rhb;

assign rhs  =   in[5]  &   in[4]  & (~in[3]) &   in[2]  &   in[1]  &   in[0];
assign cnt  =   in[5]  & (~in[4]) &   in[3]  &   in[2]  & (~in[1]) & (~in[0]);
assign rcb  = (~in[5]) & (~in[4]) &   in[3]  &   in[2]  &   in[1]  &   in[0];
assign shs  =   in[5]  &   in[4]  &   in[3]  &   in[2]  & (~in[1]) & (~in[0]);
assign lrhb = (~in[5]) &   in[4]  & (~in[3]) &   in[2]  &   in[1]  &   in[0];
assign rhb  = (~in[5]) &   in[4]  &   in[3]  &   in[2]  & (~in[1]) & (~in[0]);

endmodule  // tia_horizontal_lfsr_decoder
