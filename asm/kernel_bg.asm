; kernel_bg.asm
; being the first test of the background color fitter and our ability to
; generate assembly that can make a usable VCS kernel.

  processor 6502
  include "vcs.h"
  include "macro.h"

  SEG
  ORG $f000

Reset
StartOfFrame
  ; start of vertical blank processing
  lda #$00
  sta VBLANK
  lda #$02
  sta VSYNC

  ; 3 scanlines of VSYNC signal
  sta WSYNC
  sta WSYNC
  sta WSYNC

  ; 37 scanlines of vertical blank
  lda #$00
  sta VSYNC

; RESET STATE
  lda #$00
  ldx #$00
  ldy #$00
  sta COLUBK

  sta WSYNC   ; 1
  sta WSYNC   ; 2
  sta WSYNC   ; 3
  sta WSYNC   ; 4
  sta WSYNC   ; 5
  sta WSYNC   ; 6
  sta WSYNC   ; 7
  sta WSYNC   ; 8
  sta WSYNC   ; 9
  sta WSYNC   ; 10
  sta WSYNC   ; 11
  sta WSYNC   ; 12
  sta WSYNC   ; 13
  sta WSYNC   ; 14
  sta WSYNC   ; 15
  sta WSYNC   ; 16
  sta WSYNC   ; 17
  sta WSYNC   ; 18
  sta WSYNC   ; 19
  sta WSYNC   ; 20
  sta WSYNC   ; 21
  sta WSYNC   ; 22
  sta WSYNC   ; 23
  sta WSYNC   ; 24
  sta WSYNC   ; 25
  sta WSYNC   ; 26
  sta WSYNC   ; 27
  sta WSYNC   ; 28
  sta WSYNC   ; 29
  sta WSYNC   ; 30
  sta WSYNC   ; 31
  sta WSYNC   ; 32
  sta WSYNC   ; 33
  sta WSYNC   ; 34
  sta WSYNC   ; 35
  sta WSYNC   ; 36
  sta WSYNC   ; 37
  sta WSYNC   ; 38
  sta WSYNC   ; 39
  sta WSYNC   ; 40
  sta WSYNC   ; 41
  sta WSYNC   ; 42

;-------  begin machine-generated code

; -- scan line: 0
  lda #$f0
  sta COLUBK
  sta WSYNC

; -- scan line: 1
  sta WSYNC

; -- scan line: 2
  sta WSYNC

; -- scan line: 3
  sta WSYNC

; -- scan line: 4
  sta WSYNC

; -- scan line: 5
  sta WSYNC

; -- scan line: 6
  sta WSYNC

; -- scan line: 7
  sta WSYNC

; -- scan line: 8
  sta WSYNC

; -- scan line: 9
  sta WSYNC

; -- scan line: 10
  sta WSYNC

; -- scan line: 11
  sta WSYNC

; -- scan line: 12
  sta WSYNC

; -- scan line: 13
  sta WSYNC

; -- scan line: 14
  sta WSYNC

; -- scan line: 15
  sta WSYNC

; -- scan line: 16
  sta WSYNC

; -- scan line: 17
  ldy #$e0
  sty COLUBK
  sta WSYNC

; -- scan line: 18
  sta WSYNC

; -- scan line: 19
  sta WSYNC

; -- scan line: 20
  sta WSYNC

; -- scan line: 21
  sta WSYNC

; -- scan line: 22
  sta WSYNC

; -- scan line: 23
  sta WSYNC

; -- scan line: 24
  sta WSYNC

; -- scan line: 25
  sta WSYNC

; -- scan line: 26
  sta WSYNC

; -- scan line: 27
  sta WSYNC

; -- scan line: 28
  sta WSYNC

; -- scan line: 29
  sta WSYNC

; -- scan line: 30
  sta WSYNC

; -- scan line: 31
  sta WSYNC

; -- scan line: 32
  sta WSYNC

; -- scan line: 33
  sta WSYNC

; -- scan line: 34
  sta WSYNC

; -- scan line: 35
  sta WSYNC

; -- scan line: 36
  sta WSYNC

; -- scan line: 37
  sta WSYNC

; -- scan line: 38
  sta WSYNC

; -- scan line: 39
  sta WSYNC

; -- scan line: 40
  sta WSYNC

; -- scan line: 41
  sta WSYNC

; -- scan line: 42
  sta WSYNC

; -- scan line: 43
  sta WSYNC

; -- scan line: 44
  sta WSYNC

; -- scan line: 45
  sta WSYNC

; -- scan line: 46
  sta WSYNC

; -- scan line: 47
  sta WSYNC

; -- scan line: 48
  sta WSYNC

; -- scan line: 49
  sta WSYNC

; -- scan line: 50
  sta WSYNC

; -- scan line: 51
  sta WSYNC

; -- scan line: 52
  sta WSYNC

; -- scan line: 53
  sta WSYNC

; -- scan line: 54
  sta WSYNC

; -- scan line: 55
  sta WSYNC

; -- scan line: 56
  sta WSYNC

; -- scan line: 57
  sta WSYNC

; -- scan line: 58
  sta WSYNC

; -- scan line: 59
  sta WSYNC

; -- scan line: 60
  sta WSYNC

; -- scan line: 61
  sta WSYNC

; -- scan line: 62
  sta WSYNC

; -- scan line: 63
  sta WSYNC

; -- scan line: 64
  stx COLUBK
  sta WSYNC

; -- scan line: 65
  sta WSYNC

; -- scan line: 66
  sta WSYNC

; -- scan line: 67
  sta WSYNC

; -- scan line: 68
  sta WSYNC

; -- scan line: 69
  sta WSYNC

; -- scan line: 70
  sta WSYNC

; -- scan line: 71
  sta WSYNC

; -- scan line: 72
  sta WSYNC

; -- scan line: 73
  sta WSYNC

; -- scan line: 74
  sta WSYNC

; -- scan line: 75
  sta WSYNC

; -- scan line: 76
  sta WSYNC

; -- scan line: 77
  sta WSYNC

; -- scan line: 78
  sta WSYNC

; -- scan line: 79
  sta WSYNC

; -- scan line: 80
  sta WSYNC

; -- scan line: 81
  sta WSYNC

; -- scan line: 82
  sta WSYNC

; -- scan line: 83
  sta WSYNC

; -- scan line: 84
  sta WSYNC

; -- scan line: 85
  sta WSYNC

; -- scan line: 86
  sta WSYNC

; -- scan line: 87
  sty COLUBK
  sta WSYNC

; -- scan line: 88
  sta WSYNC

; -- scan line: 89
  sta WSYNC

; -- scan line: 90
  sta WSYNC

; -- scan line: 91
  sta WSYNC

; -- scan line: 92
  sta WSYNC

; -- scan line: 93
  sta WSYNC

; -- scan line: 94
  sta WSYNC

; -- scan line: 95
  sta WSYNC

; -- scan line: 96
  sta WSYNC

; -- scan line: 97
  sta WSYNC

; -- scan line: 98
  sta WSYNC

; -- scan line: 99
  sta WSYNC

; -- scan line: 100
  sta WSYNC

; -- scan line: 101
  sta WSYNC

; -- scan line: 102
  sta WSYNC

; -- scan line: 103
  sta WSYNC

; -- scan line: 104
  sta WSYNC

; -- scan line: 105
  sta WSYNC

; -- scan line: 106
  sta WSYNC

; -- scan line: 107
  sta WSYNC

; -- scan line: 108
  sta WSYNC

; -- scan line: 109
  sta WSYNC

; -- scan line: 110
  sta WSYNC

; -- scan line: 111
  sta WSYNC

; -- scan line: 112
  stx COLUBK
  sta WSYNC

; -- scan line: 113
  sta WSYNC

; -- scan line: 114
  sta WSYNC

; -- scan line: 115
  sta WSYNC

; -- scan line: 116
  sta WSYNC

; -- scan line: 117
  sta WSYNC

; -- scan line: 118
  sty COLUBK
  sta WSYNC

; -- scan line: 119
  sta WSYNC

; -- scan line: 120
  sta WSYNC

; -- scan line: 121
  sta WSYNC

; -- scan line: 122
  sta WSYNC

; -- scan line: 123
  sta WSYNC

; -- scan line: 124
  sta WSYNC

; -- scan line: 125
  sta WSYNC

; -- scan line: 126
  sta WSYNC

; -- scan line: 127
  sta WSYNC

; -- scan line: 128
  sta WSYNC

; -- scan line: 129
  stx COLUBK
  sta WSYNC

; -- scan line: 130
  sta WSYNC

; -- scan line: 131
  sta WSYNC

; -- scan line: 132
  sta WSYNC

; -- scan line: 133
  sta WSYNC

; -- scan line: 134
  sty COLUBK
  sta WSYNC

; -- scan line: 135
  sta WSYNC

; -- scan line: 136
  sta WSYNC

; -- scan line: 137
  sta WSYNC

; -- scan line: 138
  stx COLUBK
  sta WSYNC

; -- scan line: 139
  sta WSYNC

; -- scan line: 140
  sta WSYNC

; -- scan line: 141
  sta WSYNC

; -- scan line: 142
  sta WSYNC

; -- scan line: 143
  sta WSYNC

; -- scan line: 144
  sta WSYNC

; -- scan line: 145
  sta WSYNC

; -- scan line: 146
  sta WSYNC

; -- scan line: 147
  sta WSYNC

; -- scan line: 148
  sta WSYNC

; -- scan line: 149
  sta WSYNC

; -- scan line: 150
  sta WSYNC

; -- scan line: 151
  sta WSYNC

; -- scan line: 152
  sta WSYNC

; -- scan line: 153
  sta WSYNC

; -- scan line: 154
  sta WSYNC

; -- scan line: 155
  sta WSYNC

; -- scan line: 156
  sta WSYNC

; -- scan line: 157
  sta WSYNC

; -- scan line: 158
  sta WSYNC

; -- scan line: 159
  sta WSYNC

; -- scan line: 160
  sta WSYNC

; -- scan line: 161
  sta WSYNC

; -- scan line: 162
  sta WSYNC

; -- scan line: 163
  sta WSYNC

; -- scan line: 164
  sta WSYNC

; -- scan line: 165
  sta WSYNC

; -- scan line: 166
  sta WSYNC

; -- scan line: 167
  sta WSYNC

; -- scan line: 168
  sta WSYNC

; -- scan line: 169
  sta WSYNC

; -- scan line: 170
  sta WSYNC

; -- scan line: 171
  sta WSYNC

; -- scan line: 172
  sta WSYNC

; -- scan line: 173
  sta WSYNC

; -- scan line: 174
  sta WSYNC

; -- scan line: 175
  sta WSYNC

; -- scan line: 176
  sta WSYNC

; -- scan line: 177
  sta WSYNC

; -- scan line: 178
  sta WSYNC

; -- scan line: 179
  sta WSYNC
;-------  end machine-generated code


  lda #%01000010
  sta VBLANK                     ; end of screen - enter blanking

  sta WSYNC   ; 1
  sta WSYNC   ; 2
  sta WSYNC   ; 3
  sta WSYNC   ; 4
  sta WSYNC   ; 5
  sta WSYNC   ; 6
  sta WSYNC   ; 7
  sta WSYNC   ; 8
  sta WSYNC   ; 9
  sta WSYNC   ; 10
  sta WSYNC   ; 11
  sta WSYNC   ; 12
  sta WSYNC   ; 13
  sta WSYNC   ; 14
  sta WSYNC   ; 15
  sta WSYNC   ; 16
  sta WSYNC   ; 17
  sta WSYNC   ; 18
  sta WSYNC   ; 19
  sta WSYNC   ; 20
  sta WSYNC   ; 21
  sta WSYNC   ; 22
  sta WSYNC   ; 23
  sta WSYNC   ; 24
  sta WSYNC   ; 25
  sta WSYNC   ; 26
  sta WSYNC   ; 27
  sta WSYNC   ; 28
  sta WSYNC   ; 29
  sta WSYNC   ; 30
  sta WSYNC   ; 31
  sta WSYNC   ; 32
  sta WSYNC   ; 33
  sta WSYNC   ; 34
  sta WSYNC   ; 35
  sta WSYNC   ; 36

  jmp StartOfFrame

  ORG $fffa
  .word Reset
  .word Reset
  .word Reset
end
