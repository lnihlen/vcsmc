`ifndef TIA_TIA_BALL_POSITION_COUNTER_V
`define TIA_TIA_BALL_POSITION_COUNTER_V

`include "tia_biphase_clock.v"
`include "tia_d1.v"
`include "tia_l.v"
`include "tia_lfsr.v"

module tia_ball_position_counter(
    // input
    motck,
    blec_bar,
    blre,
    blen,
    p1gr,
    blvd,
    blsiz2,
    blsiz1_bar,
    clkp,
    d1,
    d0,
    // output
    bl);

input motck, blec_bar, blre, blen, p1gr, blvd, blsiz2, blsiz1_bar, clkp, d1, d0;
output bl;

wire motck, blec_bar, blre, blen, p1gr, blvd, blsiz2, blsiz1_bar, clkp, d1, d0;
wire bl;

wire blck;
assign blck = (~blec_bar) | motck;

wire blre_latch, bphi1, bphi2, bqb;
tia_biphase_clock bpc(.clk(blck),
                      .r(blre),
                      .phi1(bphi1),
                      .phi2(bphi2),
                      .bqb(bqb),
                      .rl(blre_latch));

wire[5:0] out;
wire lfsr_reset;
tia_lfsr lfsr(.s1(bphi1), .s2(bphi2), .reset(lfsr_reset), .out(out));

wire err, wend, rdin;
assign err = &(out);
assign wend = out[5] & (~out[4]) & out[3] & out[2] & (~out[1]) & out[0];
assign rdin = ~(blre_latch | err | wend);

wire rdout, bldout;
tia_d1 rd1(.in(rdin), .s1(bphi1), .s2(bphi2), .out(rdout));
tia_d1 bld1(.in(rdout), .s1(bphi1), .s2(bphi2), .out(bldout));

assign lfsr_reset = ~rdout;

wire blen_bar, blen_l_out, blen_l_out_bar;
assign blen_bar = ~blen;
tia_l lblen(.in(d1), .follow(blen), .latch(blen_bar), .out(blen_l_out),
    .out_bar(blen_l_out_bar));
wire p1gr_bar, p1gr_l_out;
assign p1gr_bar = ~p1gr;
tia_l lp1gr(.in(blen_l_out_bar), .follow(p1gr), .latch(p1gr_bar),
    .out(p1gr_l_out));
wire blvd_bar, blvd_l_out_bar;
assign blvd_bar = ~blvd;
tia_l lblvd(.in(d0), .follow(blvd), .latch(blvd_bar), .out_bar(blvd_l_out_bar));

wire bgvd;
assign bgvd = ~((blen_l_out & blvd_l_out_bar) | ~(p1gr_l_out | blvd_l_out_bar));

wire bf2in, bf2in_bar;
assign bf2in = ~(~(~blsiz2 | blsiz1_bar | bldout | bgvd) |
    ~(bgvd | ~(blsiz2 | ~(blsiz1_bar | bqb) | bphi2) | rdout));
assign bf2in_bar = ~bf2in;

tia_f1 bf1(.s(bf2in), .r(bf2in_bar), .clock(clkp), .reset(0), .q(bl));

endmodule  // tia_ball_position_counter

`endif  // TIA_TIA_BALL_POSITION_COUNTER_V
