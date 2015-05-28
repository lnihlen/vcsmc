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
  sta WSYNC

  ; 2 / 192
  sta WSYNC

  ; 3 / 192
  sta WSYNC

  ; 4 / 192
  sta WSYNC

  ; 5 / 192
  sta WSYNC

  ; 6 / 192
  sta WSYNC

  ; 7 / 192
  sta WSYNC

  ; 8 / 192
  sta WSYNC

  ; 9 / 192
  sta WSYNC

  ; 10 / 192
  sta WSYNC

  ; 11 / 192
  sta WSYNC

  ; 12 / 192
  sta WSYNC

  ; 13 / 192
  sta WSYNC

  ; 14 / 192
  sta WSYNC

  ; 15 / 192
  sta WSYNC

  ; 16 / 192
  sta WSYNC

  ; 17 / 192
  sta WSYNC

  ; 18 / 192
  sta WSYNC

  ; 19 / 192
  sta WSYNC

  ; 20 / 192
  sta WSYNC

  ; 21 / 192
  sta WSYNC

  ; 22 / 192
  sta WSYNC

  ; 23 / 192
  sta WSYNC

  ; 24 / 192
  sta WSYNC

  ; 25 / 192
  sta WSYNC

  ; 26 / 192
  sta WSYNC

  ; 27 / 192
  sta WSYNC

  ; 28 / 192
  sta WSYNC

  ; 29 / 192
  sta WSYNC

  ; 30 / 192
  sta WSYNC

  ; 31 / 192
  sta WSYNC

  ; 32 / 192
  sta WSYNC

  ; 33 / 192
  sta WSYNC

  ; 34 / 192
  sta WSYNC

  ; 35 / 192
  sta WSYNC

  ; 36 / 192
  sta WSYNC

  ; 37 / 192
  sta WSYNC

  ; 38 / 192
  sta WSYNC

  ; 39 / 192
  sta WSYNC

  ; 40 / 192
  sta WSYNC

  ; 41 / 192
  sta WSYNC

  ; 42 / 192
  sta WSYNC

  ; 43 / 192
  sta WSYNC

  ; 44 / 192
  sta WSYNC

  ; 45 / 192
  sta WSYNC

  ; 46 / 192
  sta WSYNC

  ; 47 / 192
  sta WSYNC

  ; 48 / 192
  sta WSYNC

  ; 49 / 192
  sta WSYNC

  ; 50 / 192
  sta WSYNC

  ; 51 / 192
  sta WSYNC

  ; 52 / 192
  sta WSYNC

  ; 53 / 192
  sta WSYNC

  ; 54 / 192
  sta WSYNC

  ; 55 / 192
  sta WSYNC

  ; 56 / 192
  sta WSYNC

  ; 57 / 192
  sta WSYNC

  ; 58 / 192
  sta WSYNC

  ; 59 / 192
  sta WSYNC

  ; 60 / 192
  sta WSYNC

  ; 61 / 192
  sta WSYNC

  ; 62 / 192
  sta WSYNC

  ; 63 / 192
  sta WSYNC

  ; 64 / 192
  sta WSYNC

  ; 65 / 192
  sta WSYNC

  ; 66 / 192
  sta WSYNC

  ; 67 / 192
  sta WSYNC

  ; 68 / 192
  sta WSYNC

  ; 69 / 192
  sta WSYNC

  ; 70 / 192
  sta WSYNC

  ; 71 / 192
  sta WSYNC

  ; 72 / 192
  sta WSYNC

  ; 73 / 192
  sta WSYNC

  ; 74 / 192
  sta WSYNC

  ; 75 / 192
  sta WSYNC

  ; 76 / 192
  sta WSYNC

  ; 77 / 192
  sta WSYNC

  ; 78 / 192
  sta WSYNC

  ; 79 / 192
  sta WSYNC

  ; 80 / 192
  sta WSYNC

  ; 81 / 192
  sta WSYNC

  ; 82 / 192
  sta WSYNC

  ; 83 / 192
  sta WSYNC

  ; 84 / 192
  sta WSYNC

  ; 85 / 192
  sta WSYNC

  ; 86 / 192
  sta WSYNC

  ; 87 / 192
  sta WSYNC

  ; 88 / 192
  sta WSYNC

  ; 89 / 192
  sta WSYNC

  ; 90 / 192
  sta WSYNC

  ; 91 / 192
  sta WSYNC

  ; 92 / 192
  sta WSYNC

  ; 93 / 192
  sta WSYNC

  ; 94 / 192
  sta WSYNC

  ; 95 / 192
  sta WSYNC

  ; 96 / 192
  sta WSYNC

  ; 97 / 192
  sta WSYNC

  ; 98 / 192
  sta WSYNC

  ; 99 / 192
  sta WSYNC

  ; 100 / 192
  sta WSYNC

  ; 101 / 192
  sta WSYNC

  ; 102 / 192
  sta WSYNC

  ; 103 / 192
  sta WSYNC

  ; 104 / 192
  sta WSYNC

  ; 105 / 192
  sta WSYNC

  ; 106 / 192
  sta WSYNC

  ; 107 / 192
  sta WSYNC

  ; 108 / 192
  sta WSYNC

  ; 109 / 192
  sta WSYNC

  ; 110 / 192
  sta WSYNC

  ; 111 / 192
  sta WSYNC

  ; 112 / 192
  sta WSYNC

  ; 113 / 192
  sta WSYNC

  ; 114 / 192
  sta WSYNC

  ; 115 / 192
  sta WSYNC

  ; 116 / 192
  sta WSYNC

  ; 117 / 192
  sta WSYNC

  ; 118 / 192
  sta WSYNC

  ; 119 / 192
  sta WSYNC

  ; 120 / 192
  sta WSYNC

  ; 121 / 192
  sta WSYNC

  ; 122 / 192
  sta WSYNC

  ; 123 / 192
  sta WSYNC

  ; 124 / 192
  sta WSYNC

  ; 125 / 192
  sta WSYNC

  ; 126 / 192
  sta WSYNC

  ; 127 / 192
  sta WSYNC

  ; 128 / 192
  sta WSYNC

  ; 129 / 192
  sta WSYNC

  ; 130 / 192
  sta WSYNC

  ; 131 / 192
  sta WSYNC

  ; 132 / 192
  sta WSYNC

  ; 133 / 192
  sta WSYNC

  ; 134 / 192
  sta WSYNC

  ; 135 / 192
  sta WSYNC

  ; 136 / 192
  sta WSYNC

  ; 137 / 192
  sta WSYNC

  ; 138 / 192
  sta WSYNC

  ; 139 / 192
  sta WSYNC

  ; 140 / 192
  sta WSYNC

  ; 141 / 192
  sta WSYNC

  ; 142 / 192
  sta WSYNC

  ; 143 / 192
  sta WSYNC

  ; 144 / 192
  sta WSYNC

  ; 145 / 192
  sta WSYNC

  ; 146 / 192
  sta WSYNC

  ; 147 / 192
  sta WSYNC

  ; 148 / 192
  sta WSYNC

  ; 149 / 192
  sta WSYNC

  ; 150 / 192
  sta WSYNC

  ; 151 / 192
  sta WSYNC

  ; 152 / 192
  sta WSYNC

  ; 153 / 192
  sta WSYNC

  ; 154 / 192
  sta WSYNC

  ; 155 / 192
  sta WSYNC

  ; 156 / 192
  sta WSYNC

  ; 157 / 192
  sta WSYNC

  ; 158 / 192
  sta WSYNC

  ; 159 / 192
  sta WSYNC

  ; 160 / 192
  sta WSYNC

  ; 161 / 192
  sta WSYNC

  ; 162 / 192
  sta WSYNC

  ; 163 / 192
  sta WSYNC

  ; 164 / 192
  sta WSYNC

  ; 165 / 192
  sta WSYNC

  ; 166 / 192
  sta WSYNC

  ; 167 / 192
  sta WSYNC

  ; 168 / 192
  sta WSYNC

  ; 169 / 192
  sta WSYNC

  ; 170 / 192
  sta WSYNC

  ; 171 / 192
  sta WSYNC

  ; 172 / 192
  sta WSYNC

  ; 173 / 192
  sta WSYNC

  ; 174 / 192
  sta WSYNC

  ; 175 / 192
  sta WSYNC

  ; 176 / 192
  sta WSYNC

  ; 177 / 192
  sta WSYNC

  ; 178 / 192
  sta WSYNC

  ; 179 / 192
  sta WSYNC

  ; 180 / 192
  sta WSYNC

  ; 181 / 192
  sta WSYNC

  ; 182 / 192
  sta WSYNC

  ; 183 / 192
  sta WSYNC

  ; 184 / 192
  sta WSYNC

  ; 185 / 192
  sta WSYNC

  ; 186 / 192
  sta WSYNC

  ; 187 / 192
  sta WSYNC

  ; 188 / 192
  sta WSYNC

  ; 189 / 192
  sta WSYNC

  ; 190 / 192
  sta WSYNC

  ; 191 / 192
  sta WSYNC

  ; 192 / 192
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
