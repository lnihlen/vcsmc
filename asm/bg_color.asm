  lda #0
  sta VBLANK
  lda #2
  sta VSYNC

  sta WSYNC  ; 3 scanlines of vsync
  sta WSYNC
  sta WSYNC

  lda #0
  sta VSYNC

  sta WSYNC  ; 35 lines (plus the two below) for 37 lines of VBlank
  sta WSYNC  ; 2/35
  sta WSYNC  ; 3/35
  sta WSYNC  ; 4/35
  sta WSYNC  ; 5/35
  sta WSYNC  ; 6/35
  sta WSYNC  ; 7/35
  sta WSYNC  ; 8/35
  sta WSYNC  ; 9/35
  sta WSYNC  ; 10/35
  sta WSYNC  ; 11/35
  sta WSYNC  ; 12/35
  sta WSYNC  ; 13/35
  sta WSYNC  ; 14/35
  sta WSYNC  ; 15/35
  sta WSYNC  ; 16/35
  sta WSYNC  ; 17/35
  sta WSYNC  ; 18/35
  sta WSYNC  ; 19/35
  sta WSYNC  ; 20/35
  sta WSYNC  ; 21/35
  sta WSYNC  ; 22/35
  sta WSYNC  ; 23/35
  sta WSYNC  ; 24/35
  sta WSYNC  ; 25/35
  sta WSYNC  ; 26/35
  sta WSYNC  ; 27/35
  sta WSYNC  ; 28/35
  sta WSYNC  ; 29/35
  sta WSYNC  ; 30/35
  sta WSYNC  ; 31/35
  sta WSYNC  ; 32/35
  sta WSYNC  ; 33/35
  sta WSYNC  ; 34/35
  sta WSYNC  ; 35/35

  lda #0
  sta RESP0
  sta RESP1
  sta NUSIZ0
  sta NUSIZ1
  sta COLUP0
  sta COLUP1
  sta COLUBK
  sta CTRLPF
  sta REFP0
  sta REFP1
  sta PF0
  sta PF1
  sta PF2
  sta AUDC0
  sta AUDC1
  sta AUDF0
  sta AUDF1
  sta AUDV0
  sta AUDV1
  sta GRP0
  sta GRP1
  sta ENAM0
  sta ENAM1
  nop

  sta ENABL
  sta VDELP0
  sta VDELP1
  sta RESMP0
  sta RESMP1
  sta HMCLR
  ldx #0
  ldy #0
  sta WSYNC

  ; 192 lines of content

  ; 1 / 192
  lda #0
  sta COLUBK
  sta WSYNC

  ; 2 / 192
  lda #2
  sta COLUBK
  sta WSYNC

  ; 3 / 192
  lda #4
  sta COLUBK
  sta WSYNC

  ; 4 / 192
  lda #6
  sta COLUBK
  sta WSYNC

  ; 5 / 192
  lda #8
  sta COLUBK
  sta WSYNC

  ; 6 / 192
  lda #10
  sta COLUBK
  sta WSYNC

  ; 7 / 192
  lda #12
  sta COLUBK
  sta WSYNC

  ; 8 / 192
  lda #14
  sta COLUBK
  sta WSYNC

  ; 9 / 192
  lda #16
  sta COLUBK
  sta WSYNC

  ; 10 / 192
  lda #18
  sta COLUBK
  sta WSYNC

  ; 11 / 192
  lda #20
  sta COLUBK
  sta WSYNC

  ; 12 / 192
  lda #22
  sta COLUBK
  sta WSYNC

  ; 13 / 192
  lda #24
  sta COLUBK
  sta WSYNC

  ; 14 / 192
  lda #26
  sta COLUBK
  sta WSYNC

  ; 15 / 192
  lda #28
  sta COLUBK
  sta WSYNC

  ; 16 / 192
  lda #30
  sta COLUBK
  sta WSYNC

  ; 17 / 192
  lda #32
  sta COLUBK
  sta WSYNC

  ; 18 / 192
  lda #34
  sta COLUBK
  sta WSYNC

  ; 19 / 192
  lda #36
  sta COLUBK
  sta WSYNC

  ; 20 / 192
  lda #38
  sta COLUBK
  sta WSYNC

  ; 21 / 192
  lda #40
  sta COLUBK
  sta WSYNC

  ; 22 / 192
  lda #42
  sta COLUBK
  sta WSYNC

  ; 23 / 192
  lda #44
  sta COLUBK
  sta WSYNC

  ; 24 / 192
  lda #46
  sta COLUBK
  sta WSYNC

  ; 25 / 192
  lda #48
  sta COLUBK
  sta WSYNC

  ; 26 / 192
  lda #50
  sta COLUBK
  sta WSYNC

  ; 27 / 192
  lda #52
  sta COLUBK
  sta WSYNC

  ; 28 / 192
  lda #54
  sta COLUBK
  sta WSYNC

  ; 29 / 192
  lda #56
  sta COLUBK
  sta WSYNC

  ; 30 / 192
  lda #58
  sta COLUBK
  sta WSYNC

  ; 31 / 192
  lda #60
  sta COLUBK
  sta WSYNC

  ; 32 / 192
  lda #62
  sta COLUBK
  sta WSYNC

  ; 33 / 192
  lda #64
  sta COLUBK
  sta WSYNC

  ; 34 / 192
  lda #66
  sta COLUBK
  sta WSYNC

  ; 35 / 192
  lda #68
  sta COLUBK
  sta WSYNC

  ; 36 / 192
  lda #70
  sta COLUBK
  sta WSYNC

  ; 37 / 192
  lda #72
  sta COLUBK
  sta WSYNC

  ; 38 / 192
  lda #74
  sta COLUBK
  sta WSYNC

  ; 39 / 192
  lda #76
  sta COLUBK
  sta WSYNC

  ; 40 / 192
  lda #78
  sta COLUBK
  sta WSYNC

  ; 41 / 192
  lda #80
  sta COLUBK
  sta WSYNC

  ; 42 / 192
  lda #82
  sta COLUBK
  sta WSYNC

  ; 43 / 192
  lda #84
  sta COLUBK
  sta WSYNC

  ; 44 / 192
  lda #86
  sta COLUBK
  sta WSYNC

  ; 45 / 192
  lda #88
  sta COLUBK
  sta WSYNC

  ; 46 / 192
  lda #90
  sta COLUBK
  sta WSYNC

  ; 47 / 192
  lda #92
  sta COLUBK
  sta WSYNC

  ; 48 / 192
  lda #94
  sta COLUBK
  sta WSYNC

  ; 49 / 192
  lda #96
  sta COLUBK
  sta WSYNC

  ; 50 / 192
  lda #98
  sta COLUBK
  sta WSYNC

  ; 51 / 192
  lda #100
  sta COLUBK
  sta WSYNC

  ; 52 / 192
  lda #102
  sta COLUBK
  sta WSYNC

  ; 53 / 192
  lda #104
  sta COLUBK
  sta WSYNC

  ; 54 / 192
  lda #106
  sta COLUBK
  sta WSYNC

  ; 55 / 192
  lda #108
  sta COLUBK
  sta WSYNC

  ; 56 / 192
  lda #110
  sta COLUBK
  sta WSYNC

  ; 57 / 192
  lda #112
  sta COLUBK
  sta WSYNC

  ; 58 / 192
  lda #114
  sta COLUBK
  sta WSYNC

  ; 59 / 192
  lda #116
  sta COLUBK
  sta WSYNC

  ; 60 / 192
  lda #118
  sta COLUBK
  sta WSYNC

  ; 61 / 192
  lda #120
  sta COLUBK
  sta WSYNC

  ; 62 / 192
  lda #122
  sta COLUBK
  sta WSYNC

  ; 63 / 192
  lda #124
  sta COLUBK
  sta WSYNC

  ; 64 / 192
  lda #126
  sta COLUBK
  sta WSYNC

  ; 65 / 192
  lda #128
  sta COLUBK
  sta WSYNC

  ; 66 / 192
  lda #130
  sta COLUBK
  sta WSYNC

  ; 67 / 192
  lda #132
  sta COLUBK
  sta WSYNC

  ; 68 / 192
  lda #134
  sta COLUBK
  sta WSYNC

  ; 69 / 192
  lda #136
  sta COLUBK
  sta WSYNC

  ; 70 / 192
  lda #138
  sta COLUBK
  sta WSYNC

  ; 71 / 192
  lda #140
  sta COLUBK
  sta WSYNC

  ; 72 / 192
  lda #142
  sta COLUBK
  sta WSYNC

  ; 73 / 192
  lda #144
  sta COLUBK
  sta WSYNC

  ; 74 / 192
  lda #146
  sta COLUBK
  sta WSYNC

  ; 75 / 192
  lda #148
  sta COLUBK
  sta WSYNC

  ; 76 / 192
  lda #150
  sta COLUBK
  sta WSYNC

  ; 77 / 192
  lda #152
  sta COLUBK
  sta WSYNC

  ; 78 / 192
  lda #154
  sta COLUBK
  sta WSYNC

  ; 79 / 192
  lda #156
  sta COLUBK
  sta WSYNC

  ; 80 / 192
  lda #158
  sta COLUBK
  sta WSYNC

  ; 81 / 192
  lda #160
  sta COLUBK
  sta WSYNC

  ; 82 / 192
  lda #162
  sta COLUBK
  sta WSYNC

  ; 83 / 192
  lda #164
  sta COLUBK
  sta WSYNC

  ; 84 / 192
  lda #166
  sta COLUBK
  sta WSYNC

  ; 85 / 192
  lda #168
  sta COLUBK
  sta WSYNC

  ; 86 / 192
  lda #170
  sta COLUBK
  sta WSYNC

  ; 87 / 192
  lda #172
  sta COLUBK
  sta WSYNC

  ; 88 / 192
  lda #174
  sta COLUBK
  sta WSYNC

  ; 89 / 192
  lda #176
  sta COLUBK
  sta WSYNC

  ; 90 / 192
  lda #178
  sta COLUBK
  sta WSYNC

  ; 91 / 192
  lda #180
  sta COLUBK
  sta WSYNC

  ; 92 / 192
  lda #182
  sta COLUBK
  sta WSYNC

  ; 93 / 192
  lda #184
  sta COLUBK
  sta WSYNC

  ; 94 / 192
  lda #186
  sta COLUBK
  sta WSYNC

  ; 95 / 192
  lda #188
  sta COLUBK
  sta WSYNC

  ; 96 / 192
  lda #190
  sta COLUBK
  sta WSYNC

  ; 97 / 192
  lda #192
  sta COLUBK
  sta WSYNC

  ; 98 / 192
  lda #194
  sta COLUBK
  sta WSYNC

  ; 99 / 192
  lda #196
  sta COLUBK
  sta WSYNC

  ; 100 / 192
  lda #198
  sta COLUBK
  sta WSYNC

  ; 101 / 192
  lda #200
  sta COLUBK
  sta WSYNC

  ; 102 / 192
  lda #202
  sta COLUBK
  sta WSYNC

  ; 103 / 192
  lda #204
  sta COLUBK
  sta WSYNC

  ; 104 / 192
  lda #206
  sta COLUBK
  sta WSYNC

  ; 105 / 192
  lda #208
  sta COLUBK
  sta WSYNC

  ; 106 / 192
  lda #210
  sta COLUBK
  sta WSYNC

  ; 107 / 192
  lda #212
  sta COLUBK
  sta WSYNC

  ; 108 / 192
  lda #214
  sta COLUBK
  sta WSYNC

  ; 109 / 192
  lda #216
  sta COLUBK
  sta WSYNC

  ; 110 / 192
  lda #218
  sta COLUBK
  sta WSYNC

  ; 111 / 192
  lda #220
  sta COLUBK
  sta WSYNC

  ; 112 / 192
  lda #222
  sta COLUBK
  sta WSYNC

  ; 113 / 192
  lda #224
  sta COLUBK
  sta WSYNC

  ; 114 / 192
  lda #226
  sta COLUBK
  sta WSYNC

  ; 115 / 192
  lda #228
  sta COLUBK
  sta WSYNC

  ; 116 / 192
  lda #230
  sta COLUBK
  sta WSYNC

  ; 117 / 192
  lda #232
  sta COLUBK
  sta WSYNC

  ; 118 / 192
  lda #234
  sta COLUBK
  sta WSYNC

  ; 119 / 192
  lda #236
  sta COLUBK
  sta WSYNC

  ; 120 / 192
  lda #238
  sta COLUBK
  sta WSYNC

  ; 121 / 192
  lda #240
  sta COLUBK
  sta WSYNC

  ; 122 / 192
  lda #242
  sta COLUBK
  sta WSYNC

  ; 123 / 192
  lda #246
  sta COLUBK
  sta WSYNC

  ; 124 / 192
  lda #248
  sta COLUBK
  sta WSYNC

  ; 125 / 192
  lda #250
  sta COLUBK
  sta WSYNC

  ; 126 / 192
  lda #252
  sta COLUBK
  sta WSYNC

  ; 127 / 192
  lda #254
  sta COLUBK
  sta WSYNC

  ; 128 / 192
  lda #0
  sta COLUBK
  sta WSYNC

  ; 129 / 192
  lda #2
  sta COLUBK
  sta WSYNC

  ; 130 / 192
  lda #4
  sta COLUBK
  sta WSYNC

  ; 131 / 192
  lda #6
  sta COLUBK
  sta WSYNC

  ; 132 / 192
  lda #8
  sta COLUBK
  sta WSYNC

  ; 133 / 192
  lda #10
  sta COLUBK
  sta WSYNC

  ; 134 / 192
  lda #12
  sta COLUBK
  sta WSYNC

  ; 135 / 192
  lda #14
  sta COLUBK
  sta WSYNC

  ; 136 / 192
  lda #16
  sta COLUBK
  sta WSYNC

  ; 137 / 192
  lda #18
  sta COLUBK
  sta WSYNC

  ; 138 / 192
  lda #20
  sta COLUBK
  sta WSYNC

  ; 139 / 192
  lda #22
  sta COLUBK
  sta WSYNC

  ; 140 / 192
  lda #24
  sta COLUBK
  sta WSYNC

  ; 141 / 192
  lda #26
  sta COLUBK
  sta WSYNC

  ; 142 / 192
  lda #28
  sta COLUBK
  sta WSYNC

  ; 143 / 192
  lda #30
  sta COLUBK
  sta WSYNC

  ; 144 / 192
  lda #32
  sta COLUBK
  sta WSYNC

  ; 145 / 192
  lda #34
  sta COLUBK
  sta WSYNC

  ; 146 / 192
  lda #36
  sta COLUBK
  sta WSYNC

  ; 147 / 192
  lda #38
  sta COLUBK
  sta WSYNC

  ; 148 / 192
  lda #40
  sta COLUBK
  sta WSYNC

  ; 149 / 192
  lda #42
  sta COLUBK
  sta WSYNC

  ; 150 / 192
  lda #44
  sta COLUBK
  sta WSYNC

  ; 151 / 192
  lda #46
  sta COLUBK
  sta WSYNC

  ; 152 / 192
  lda #48
  sta COLUBK
  sta WSYNC

  ; 153 / 192
  lda #50
  sta COLUBK
  sta WSYNC

  ; 154 / 192
  lda #52
  sta COLUBK
  sta WSYNC

  ; 155 / 192
  lda #54
  sta COLUBK
  sta WSYNC

  ; 156 / 192
  lda #56
  sta COLUBK
  sta WSYNC

  ; 157 / 192
  lda #58
  sta COLUBK
  sta WSYNC

  ; 158 / 192
  lda #60
  sta COLUBK
  sta WSYNC

  ; 159 / 192
  lda #62
  sta COLUBK
  sta WSYNC

  ; 160 / 192
  lda #64
  sta COLUBK
  sta WSYNC

  ; 161 / 192
  lda #66
  sta COLUBK
  sta WSYNC

  ; 162 / 192
  lda #68
  sta COLUBK
  sta WSYNC

  ; 163 / 192
  lda #70
  sta COLUBK
  sta WSYNC

  ; 164 / 192
  lda #72
  sta COLUBK
  sta WSYNC

  ; 165 / 192
  lda #74
  sta COLUBK
  sta WSYNC

  ; 166 / 192
  lda #76
  sta COLUBK
  sta WSYNC

  ; 167 / 192
  lda #78
  sta COLUBK
  sta WSYNC

  ; 168 / 192
  lda #80
  sta COLUBK
  sta WSYNC

  ; 169 / 192
  lda #82
  sta COLUBK
  sta WSYNC

  ; 170 / 192
  lda #84
  sta COLUBK
  sta WSYNC

  ; 171 / 192
  lda #86
  sta COLUBK
  sta WSYNC

  ; 172 / 192
  lda #88
  sta COLUBK
  sta WSYNC

  ; 173 / 192
  lda #90
  sta COLUBK
  sta WSYNC

  ; 174 / 192
  lda #92
  sta COLUBK
  sta WSYNC

  ; 175 / 192
  lda #94
  sta COLUBK
  sta WSYNC

  ; 176 / 192
  lda #96
  sta COLUBK
  sta WSYNC

  ; 177 / 192
  lda #98
  sta COLUBK
  sta WSYNC

  ; 178 / 192
  lda #100
  sta COLUBK
  sta WSYNC

  ; 179 / 192
  lda #102
  sta COLUBK
  sta  WSYNC

  ; 180 / 192
  lda #104
  sta COLUBK
  sta WSYNC

  ; 181 / 192
  lda #106
  sta COLUBK
  sta WSYNC

  ; 182 / 192
  lda #108
  sta COLUBK
  sta WSYNC

  ; 183 / 192
  lda #110
  sta COLUBK
  sta WSYNC

  ; 184 / 192
  lda #112
  sta COLUBK
  sta WSYNC

  ; 185 / 192
  lda #114
  sta COLUBK
  sta WSYNC

  ; 186 / 192
  lda #116
  sta COLUBK
  sta WSYNC

  ; 187 / 192
  lda #118
  sta COLUBK
  sta WSYNC

  ; 188 / 192
  lda #120
  sta COLUBK
  sta WSYNC

  ; 189 / 192
  lda #122
  sta COLUBK
  sta WSYNC

  ; 190 / 192
  lda #124
  sta COLUBK
  sta WSYNC

  ; 191 / 192
  lda #126
  sta COLUBK
  sta WSYNC

  ; 192 / 192
  lda #128
  sta COLUBK
  sta WSYNC

  ; turn on vblank, then 30 scanlines of overscan
  lda #2
  sta VBLANK

  sta WSYNC  ; 1 / 30
  sta WSYNC  ; 2 / 30
  sta WSYNC  ; 3 / 30
  sta WSYNC  ; 4 / 30
  sta WSYNC  ; 5 / 30
  sta WSYNC  ; 6 / 30
  sta WSYNC  ; 7 / 30
  sta WSYNC  ; 8 / 30
  sta WSYNC  ; 9 / 30
  sta WSYNC  ; 10 / 30
  sta WSYNC  ; 11 / 30
  sta WSYNC  ; 12 / 30
  sta WSYNC  ; 13 / 30
  sta WSYNC  ; 14 / 30
  sta WSYNC  ; 15 / 30
  sta WSYNC  ; 16 / 30
  sta WSYNC  ; 17 / 30
  sta WSYNC  ; 18 / 30
  sta WSYNC  ; 19 / 30
  sta WSYNC  ; 20 / 30
  sta WSYNC  ; 21 / 30
  sta WSYNC  ; 22 / 30
  sta WSYNC  ; 23 / 30
  sta WSYNC  ; 24 / 30
  sta WSYNC  ; 25 / 30
  sta WSYNC  ; 26 / 30
  sta WSYNC  ; 27 / 30
  sta WSYNC  ; 28 / 30
  sta WSYNC  ; 29 / 30
  sta WSYNC  ; 30 / 30
  jmp $0800
