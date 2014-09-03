; tries to update the sample 2x a scanline, at 0 clocks and at 38 clocks

  processor 6502
  include "vcs.h"
  include "macro.h"

FFS EQU $1000
PCM0FFS EQU #0

  SEG
  ORG $f000

Reset
  ; initialize audio system for mono PCM
  lda #0
  sta AUDC0
  sta AUDC1
  sta AUDV0
  sta AUDV1

  ; turn on fast fetch streaming
  lda #0
  sta FFS

StartOfFrame

  ; ------- 3 scanlines of vertical sync
  ldx #2
  stx VSYNC
  ldy #3
VerticalSync
  stx WSYNC
  lda PCM0FFS   ; 00/76 ; fast load channel 0
  sta AUDV0     ; 02/76
  sta COLUP0    ; 05/76 ; 3-cycle no-op
  nop           ; 08/76
  nop           ; 10/76
  nop           ; 12/76
  nop           ; 14/76
  nop           ; 16/76
  nop           ; 18/76
  nop           ; 20/76
  nop           ; 22/76
  nop           ; 24/76
  nop           ; 26/76
  nop           ; 28/76
  nop           ; 30/76
  nop           ; 32/76
  nop           ; 34/76
  nop           ; 36/76
  lda PCM0FFS   ; 38/76
  sta AUDV0     ; 40/76
  dey
  bne VerticalSync

  ; ------- 37 scanlines of vertical blank
  ldx #0
  stx VSYNC
  ldx #2
  stx VBLANK
  ldy #37
VerticalBlank
  stx WSYNC
  lda PCM0FFS   ; 00/76 ; fast load channel 0
  sta AUDV0     ; 02/76
  sta COLUP0    ; 05/76 ; 3-cycle no-op
  nop           ; 08/76
  nop           ; 10/76
  nop           ; 12/76
  nop           ; 14/76
  nop           ; 16/76
  nop           ; 18/76
  nop           ; 20/76
  nop           ; 22/76
  nop           ; 24/76
  nop           ; 26/76
  nop           ; 28/76
  nop           ; 30/76
  nop           ; 32/76
  nop           ; 34/76
  nop           ; 36/76
  lda PCM0FFS   ; 38/76
  sta AUDV0     ; 40/76
  dey
  bne VerticalBlank

  ; ------- 192 scanlines of content
  ldx #0
  stx VBLANK
  ldy #192
  ldx #0
Content
  stx WSYNC
  lda PCM0FFS   ; 00/76 ; fast load channel 0
  sta AUDV0     ; 02/76
  sta COLUP0    ; 05/76 ; 3-cycle no-op
  nop           ; 08/76
  nop           ; 10/76
  nop           ; 12/76
  nop           ; 14/76
  nop           ; 16/76
  nop           ; 18/76
  nop           ; 20/76
  nop           ; 22/76
  nop           ; 24/76
  nop           ; 26/76
  nop           ; 28/76
  nop           ; 30/76
  nop           ; 32/76
  nop           ; 34/76
  nop           ; 36/76
  lda PCM0FFS   ; 38/76
  sta AUDV0     ; 40/76
  dey
  bne Content

  ; ------- 30 scanlines of overscan
  ldx #2
  stx VBLANK
  ldy #32
Overscan
  stx WSYNC
  lda PCM0FFS   ; 00/76 ; fast load channel 0
  sta AUDV0     ; 02/76
  sta COLUP0    ; 05/76 ; 3-cycle no-op
  nop           ; 08/76
  nop           ; 10/76
  nop           ; 12/76
  nop           ; 14/76
  nop           ; 16/76
  nop           ; 18/76
  nop           ; 20/76
  nop           ; 22/76
  nop           ; 24/76
  nop           ; 26/76
  nop           ; 28/76
  nop           ; 30/76
  nop           ; 32/76
  nop           ; 34/76
  nop           ; 36/76
  lda PCM0FFS   ; 38/76
  sta AUDV0     ; 40/76
  dey
  bne Overscan

  ; ------- back to top of frame
  jmp StartOfFrame

  echo "------", ($fff0 - *), "bytes of ROM left"

  ORG $fff0
  DC "VCSMC FFS"

  ORG $fffa
  .word Reset
  .word Reset
  .word Reset
end
