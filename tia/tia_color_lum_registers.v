`include "sr.v"
`include "tia_f1.v"
`include "tia_l.v"

module tia_color_lum_registers_cell(d, follow, latch, prio_bar, cm);
  input d, follow, latch, prio_bar;
  output cm;

  wire d, follow, latch, prio_bar;
  wire cm;

  wire dl;
  tia_l l(.in(d), .follow(follow), .latch(latch), .out(dl));
  assign cm = ~(dl & prio_bar);
endmodule  // tia_color_lum_registers_cell

module tia_color_lum_registers(
    // input
    p0,
    m0,
    p1,
    m1,
    pf,
    bl,
    blank,
    cntd,
    score_bar,
    pfp_bar,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    bkci,
    pfci,
    p1ci,
    p0ci,
    clkp,
    // output
    blk_bar,
    l0,
    l1,
    l2,
    c0,  // Color registers emulated, these are inputs to the color select
    c1,  // decode block on the schematics.
    c2,
    c3);

input p0, m0, p1, m1, pf, bl, blank, cntd, score_bar, pfp_bar, d1, d2, d3,
    d4, d5, d6, d7, bkci, pfci, p1ci, p0ci, clkp;

output blk_bar, l0, l1, l2, c0, c1, c2, c3;

wire p0, m0, p1, m1, pf, bl, blank, cntd, score_bar, pfp_bar, d1, d2, d3,
    d4, d5, d6, d7, bkci, pfci, p1ci, p0ci, clkp;

wire blk_bar, l0, l1, l2, c0, c1, c2, c3;

wire bcsrq, bcsrq_bar;
sr bcsr(.s(blank), .r(cntd), .r2(0), .q(bcsrq), .q_bar(bcsrq_bar));

wire pfbl;
assign pfbl = ~(pf | bl);
wire pfpblpf;
assign pfpblpf = ~(pfbl | pfp_bar);
wire pf_bar;
assign pf_bar = ~pf;

wire bkpri, pfpri, p1pri, p0pri;

assign bkpri = ~(blank | p0pri | p1pri | pfpri);
assign pfpri = ~(blank | p0pri | p1pri | pfbl);
assign p1pri = ~(blank | p0pri | pfpblpf |
    ~(p1 | m1 | ~(pf_bar | score_bar | bcsrq_bar)));
assign p0pri = ~(blank | pfpblpf | ~(p0 | m0 | ~(bcsrq | pf_bar | score_bar)));

wire bkci_bar, pfci_bar, p1ci_bar, p0ci_bar;
assign bkci_bar = ~bkci;
assign pfci_bar = ~pfci;
assign p1ci_bar = ~p1ci;
assign p0ci_bar = ~p0ci;

wire cmbk1, cmbk2, cmbk3, cmbk4, cmbk5, cmbk6, cmbk7;
tia_color_lum_registers_cell bkd1(.d(d1), .follow(bkci), .latch(bkci_bar),
    .prio_bar(bkpri), .cm(cmbk1));
tia_color_lum_registers_cell bkd2(.d(d2), .follow(bkci), .latch(bkci_bar),
    .prio_bar(bkpri), .cm(cmbk2));
tia_color_lum_registers_cell bkd3(.d(d3), .follow(bkci), .latch(bkci_bar),
    .prio_bar(bkpri), .cm(cmbk3));
tia_color_lum_registers_cell bkd4(.d(d4), .follow(bkci), .latch(bkci_bar),
    .prio_bar(bkpri), .cm(cmbk4));
tia_color_lum_registers_cell bkd5(.d(d5), .follow(bkci), .latch(bkci_bar),
    .prio_bar(bkpri), .cm(cmbk5));
tia_color_lum_registers_cell bkd6(.d(d6), .follow(bkci), .latch(bkci_bar),
    .prio_bar(bkpri), .cm(cmbk6));
tia_color_lum_registers_cell bkd7(.d(d7), .follow(bkci), .latch(bkci_bar),
    .prio_bar(bkpri), .cm(cmbk7));

wire cmpf1, cmpf2, cmpf3, cmpf4, cmpf5, cmpf6, cmpf7;
tia_color_lum_registers_cell pfd1(.d(d1), .follow(pfci), .latch(pfci_bar),
    .prio_bar(pfpri), .cm(cmpf1));
tia_color_lum_registers_cell pfd2(.d(d2), .follow(pfci), .latch(pfci_bar),
    .prio_bar(pfpri), .cm(cmpf2));
tia_color_lum_registers_cell pfd3(.d(d3), .follow(pfci), .latch(pfci_bar),
    .prio_bar(pfpri), .cm(cmpf3));
tia_color_lum_registers_cell pfd4(.d(d4), .follow(pfci), .latch(pfci_bar),
    .prio_bar(pfpri), .cm(cmpf4));
tia_color_lum_registers_cell pfd5(.d(d5), .follow(pfci), .latch(pfci_bar),
    .prio_bar(pfpri), .cm(cmpf5));
tia_color_lum_registers_cell pfd6(.d(d6), .follow(pfci), .latch(pfci_bar),
    .prio_bar(pfpri), .cm(cmpf6));
tia_color_lum_registers_cell pfd7(.d(d7), .follow(pfci), .latch(pfci_bar),
    .prio_bar(pfpri), .cm(cmpf7));

wire cmp11, cmp12, cmp13, cmp14, cmp15, cmp16, cmp17;
tia_color_lum_registers_cell p1d1(.d(d1), .follow(p1ci), .latch(p1ci_bar),
    .prio_bar(p1pri), .cm(cmp11));
tia_color_lum_registers_cell p1d2(.d(d2), .follow(p1ci), .latch(p1ci_bar),
    .prio_bar(p1pri), .cm(cmp12));
tia_color_lum_registers_cell p1d3(.d(d3), .follow(p1ci), .latch(p1ci_bar),
    .prio_bar(p1pri), .cm(cmp13));
tia_color_lum_registers_cell p1d4(.d(d4), .follow(p1ci), .latch(p1ci_bar),
    .prio_bar(p1pri), .cm(cmp14));
tia_color_lum_registers_cell p1d5(.d(d5), .follow(p1ci), .latch(p1ci_bar),
    .prio_bar(p1pri), .cm(cmp15));
tia_color_lum_registers_cell p1d6(.d(d6), .follow(p1ci), .latch(p1ci_bar),
    .prio_bar(p1pri), .cm(cmp16));
tia_color_lum_registers_cell p1d7(.d(d7), .follow(p1ci), .latch(p1ci_bar),
    .prio_bar(p1pri), .cm(cmp17));

wire cmp01, cmp02, cmp03, cmp04, cmp05, cmp06, cmp07;
tia_color_lum_registers_cell p0d1(.d(d1), .follow(p0ci), .latch(p0ci_bar),
    .prio_bar(p0pri), .cm(cmp01));
tia_color_lum_registers_cell p0d2(.d(d2), .follow(p0ci), .latch(p0ci_bar),
    .prio_bar(p0pri), .cm(cmp02));
tia_color_lum_registers_cell p0d3(.d(d3), .follow(p0ci), .latch(p0ci_bar),
    .prio_bar(p0pri), .cm(cmp03));
tia_color_lum_registers_cell p0d4(.d(d4), .follow(p0ci), .latch(p0ci_bar),
    .prio_bar(p0pri), .cm(cmp04));
tia_color_lum_registers_cell p0d5(.d(d5), .follow(p0ci), .latch(p0ci_bar),
    .prio_bar(p0pri), .cm(cmp05));
tia_color_lum_registers_cell p0d6(.d(d6), .follow(p0ci), .latch(p0ci_bar),
    .prio_bar(p0pri), .cm(cmp06));
tia_color_lum_registers_cell p0d7(.d(d7), .follow(p0ci), .latch(p0ci_bar),
    .prio_bar(p0pri), .cm(cmp07));

wire cm1_bar, cm2_bar, cm3_bar, cm4_bar, cm5_bar, cm6_bar, cm7_bar;
assign cm1_bar = cmbk1 & cmpf1 & cmp11 & cmp01;
assign cm2_bar = cmbk2 & cmpf2 & cmp12 & cmp02;
assign cm3_bar = cmbk3 & cmpf3 & cmp13 & cmp03;
assign cm4_bar = cmbk4 & cmpf4 & cmp14 & cmp04;
assign cm5_bar = cmbk5 & cmpf5 & cmp15 & cmp05;
assign cm6_bar = cmbk6 & cmpf6 & cmp16 & cmp06;
assign cm7_bar = cmbk7 & cmpf7 & cmp17 & cmp07;

wire cm1, cm2, cm3, cm4, cm5, cm6, cm7;
assign cm1 = ~cm1_bar;
assign cm2 = ~cm2_bar;
assign cm3 = ~cm3_bar;
assign cm4 = ~cm4_bar;
assign cm5 = ~cm5_bar;
assign cm6 = ~cm6_bar;
assign cm7 = ~cm7_bar;

wire blank_bar;
assign blank_bar = ~blank;

tia_f1 bf1(.s(blank), .r(blank_bar), .clock(clkp), .reset(0), .q(blk_bar));
tia_f1 cm1f1(.s(cm1_bar), .r(cm1), .clock(clkp), .reset(0), .q(l0));
tia_f1 cm2f1(.s(cm2_bar), .r(cm2), .clock(clkp), .reset(0), .q(l1));
tia_f1 cm3f1(.s(cm3_bar), .r(cm3), .clock(clkp), .reset(0), .q(l2));
tia_f1 cm4f1(.s(cm4_bar), .r(cm4), .clock(clkp), .reset(0), .q(c0));
tia_f1 cm5f1(.s(cm5_bar), .r(cm5), .clock(clkp), .reset(0), .q(c1));
tia_f1 cm6f1(.s(cm6_bar), .r(cm6), .clock(clkp), .reset(0), .q(c2));
tia_f1 cm7f1(.s(cm7_bar), .r(cm7), .clock(clkp), .reset(0), .q(c3));

endmodule  // tia_color_lum_registers
