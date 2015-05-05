`include "sr.v"
`include "tia_f1.v"
`include "tia_d1.v"
`include "tia_d2.v"
`include "tia_dl.v"
`include "tia_l.v"

module tia_horizontal_timing(
    // inputs:
    hphi1,
    hphi2,
    rhs,
    cnt,
    rcb,
    lrhb,
    rhb,
    shs,
    vsyn,
    vblk,
    rsynd,
    hmove,
    clkp,
    wsyn,
    d1,
    clk,
    shb,
    // outputs:
    resphi0,
    aphi1,
    aphi2,
    cb,
    vb,
    vb_bar,
    blank,
    motck,
    rdy,
    sec,
    syn);

input hphi1, hphi2, rhs, cnt, rcb, lrhb, rhb, shs, vsyn, vblk, rsynd, hmove,
    clkp, wsyn, d1, clk, shb;

output resphi0, aphi1, aphi2, cb, vb, vb_bar, blank, motck, rdy, sec, syn;

wire hphi1, hphi2, rhs, cnt, rcb, lrhb, rhb, shs, vsyn, vblk, rsynd, hmove,
    clkp, wsyn, d1, clk, shb;

wire resphi0, aphi1, aphi2, cb, vb, vb_bar, blank, motck, rdy, sec, syn;

tia_d2 d2_aphi2(.in1(rhs),
                .in2(cnt),
                .s1(hphi1),
                .s2(hphi2),
                .out(aphi2));

wire rhs_d;
tia_d1 d1_rhs(.in(rhs), .s1(hphi1), .s2(hphi2), .out(rhs_d));

tia_d2 d2_aphi1(.in1(shb),
                .in2(lrhb),
                .s1(hphi1),
                .s2(hphi2),
                .out(aphi1));

wire cb_or;
tia_dl dl_cb(.in(rcb), .s1(hphi1), .s2(hphi2), .r(rhs_d), .out(cb_or));

assign cb = ~(vs | cb_or);

wire hs;
tia_dl dl_hs(.in(shs), .s1(hphi1), .s2(hphi2), .r(rhs_d), .out(hs));

assign resphi0 = hphi2 & rsynd;

wire hb_q, hb_q_bar;
sr sr_hb(.s(sec), .r(shb), .r2(0), .q(hb_q), .q_bar(hb_q_bar));
wire hb_dl;
assign hb_dl = (hb_q & rhb) | (lrhb & hb_q_bar);
wire hb_bar;
tia_dl dl_hb(.in(hb_dl), .s1(hphi1), .s2(hphi2), .r(rhs_d), .out(hb_bar));
wire d1_sec_s, d1_sec_in;
sr sr_sec(.s(d1_sec_s), .r(hmove), .r2(0), .q(d1_sec_in));
tia_d1 d1_sec(.in(d1_sec_in), .s1(hphi1), .s2(hphi2), .out(sec));
assign d1_sec_s = hphi1 & sec;

wire comp_sync_bar;
assign comp_sync_bar = vs ^ hs;
assign syn = ~comp_sync_bar;

wire vsyn_bar = ~vsyn;
wire vs;
tia_l l_vs(.in(d1), .follow(vsyn), .latch(vsyn_bar), .out(vs));
wire vblk_bar = ~vblk;
tia_l l_vb(.in(d1), .follow(vblk), .latch(vblk_bar), .out(vb));
assign vb_bar = ~vb;

wire sr_rdy_s;
assign sr_rdy_s = clkp & shb;
wire sr_rdy_r;
assign sr_rdy_r = (~shb) & wsyn;
wire rdy_bar;
sr sr_rdy(.s(sr_rdy_s), .r(sr_rdy_r), .r2(0), .q(rdy_bar));
assign rdy = ~rdy_bar;

wire hb;
assign hb = ~hb_bar;
assign motck = ~(hb | clk);

wire blank_s_bar;
assign blank_s_bar = ~(vb_bar & hb_bar);
wire blank_r_bar;
assign blank_r_bar = ~blank_s_bar;
tia_f1 f1_blank(.s(blank_s_bar), .r(blank_r_bar), .clock(clkp), .reset(0),
    .q_bar(blank));

endmodule  // tia_horizontal_timing
