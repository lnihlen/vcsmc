; ref_simple.asm
; quick experiment to determine if we can set the BG color while drawing
; playfield bits, and vice-versa
;
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

  ldx #210
  ldy #$00
  lda #%00000000
  sta PF0
  lda #%01111100
  sta PF1
  lda #%11111000
  sta PF2
  cld         ; clear decimal flag to avoid BCD arithmetic
  ; 192 scanlines / 16 colors = 12 lines per color set
  sta WSYNC

ScanLine
  sta WSYNC   ; 33 or the end of every color scan line
  tya         ;  lda #$00
  sta COLUBK
  adc #$02
  sta COLUPF
  adc #$02
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  sta COLUBK
  adc #$02
  sta COLUPF
  adc #$02
  nop
  sta COLUBK
  adc #$02
  nop
  sta COLUPF
  adc #$02
  nop
  sta COLUBK
  adc #$02
  nop
  sta COLUPF
  dex
  bne ScanLine

  sta WSYNC

  lda #%01000010
  sta VBLANK                     ; end of screen - enter blanking

  ; 30 scanlines of overscan
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
  jmp StartOfFrame

  ORG $fffa
  .word Reset
  .word Reset
  .word Reset
end
