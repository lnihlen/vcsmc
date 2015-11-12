  processor 6502
  include "vcs.h"
  include "macro.h"

  SEG
  ORG $F000

Reset
StartOfFrame

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

  lda #0     ;  2/76
  sta RESP0  ;  5/76
  sta RESP1  ;  8/76
  sta NUSIZ0 ; 11/76
  sta NUSIZ1 ; 14/76
  sta COLUP0 ; 17/76
  sta COLUP1 ; 20/76
  sta COLUBK ; 23/76
  sta CTRLPF ; 26/76
  sta REFP0  ; 29/76
  sta REFP1  ; 32/76
  sta PF0    ; 35/76
  sta PF1    ; 38/76
  sta PF2    ; 41/76
  sta AUDC0  ; 44/76
  sta AUDC1  ; 47/76
  sta AUDF0  ; 50/76
  sta AUDF1  ; 53/76
  sta AUDV0  ; 56/76
  sta AUDV1  ; 59/76
  sta GRP0   ; 62/76
  sta GRP1   ; 65/76
  sta ENAM0  ; 68/76
  sta ENAM1  ; 71/76
  sta ENABL  ; 74/76
  nop        ; 76/76

  sta VDELP0
  sta VDELP1
  sta RESMP0
  sta RESMP1
  sta HMCLR
  ldx #0
  ldy #0
  lda #$e7
  sta GRP0
  lda #$0e
  sta COLUP0
  sta WSYNC

  ; 192 lines of content

  ; 1 / 192  -- wait none
  sta RESP0
  sta WSYNC

  ; 2 / 192  -- wait 2
  nop         ; 2
  sta RESP0
  sta WSYNC

  ; 3 / 192  -- wait 3
  sta COLUP0  ; 3
  sta RESP0
  sta WSYNC

  ; 4 / 192  -- wait 4
  nop         ; 2
  nop         ; 4
  sta RESP0
  sta WSYNC

  ; 5 / 192  -- wait 5
  nop         ; 2
  sta COLUP0  ; 5
  sta RESP0
  sta WSYNC

  ; 6 / 192  -- wait 6
  nop         ; 2
  nop         ; 4
  nop         ; 6
  sta RESP0
  sta WSYNC

  ; 7 / 192  -- wait 7
  nop         ; 2
  nop         ; 4
  sta COLUP0  ; 7
  sta RESP0
  sta WSYNC

  ; 8 / 192  -- wait 8
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  sta RESP0
  sta WSYNC

  ; 9 / 192  -- wait 9
  nop         ; 2
  nop         ; 4
  nop         ; 6
  sta COLUP0  ; 9
  sta RESP0
  sta WSYNC

  ; 10 / 192  -- wait 10
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  sta RESP0
  sta WSYNC

  ; 11 / 192  -- wait 11
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  sta COLUP0  ; 11
  sta RESP0
  sta WSYNC

  ; 12 / 192  -- wait 12
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  sta RESP0
  sta WSYNC

  ; 13 / 192  -- wait 13
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  sta COLUP0  ; 13
  sta RESP0
  sta WSYNC

  ; 14 / 192  -- wait 14
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  sta COLUP0
  sta RESP0
  sta WSYNC

  ; 15 / 192  -- wait 15
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  sta COLUP0  ; 15
  sta RESP0
  sta WSYNC

  ; 16 / 192  -- wait 16
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  sta RESP0
  sta WSYNC

  ; 17 / 192  -- wait 17
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  sta COLUP0  ; 17
  sta RESP0
  sta WSYNC

  ; 18 / 192  -- wait 18
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  sta RESP0
  sta WSYNC

  ; 19 / 192  -- wait 19
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  sta COLUP0  ; 19
  sta RESP0
  sta WSYNC

  ; 20 / 192  -- wait 20
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  sta RESP0
  sta WSYNC

  ; 21 / 192  -- wait 21
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  sta COLUP0  ; 21
  sta RESP0
  sta WSYNC

  ; 22 / 192  -- wait 22
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  sta RESP0
  sta WSYNC

  ; 23 / 192  -- wait 23
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  sta COLUP0  ; 23
  sta RESP0
  sta WSYNC

  ; 24 / 192  -- wait 24
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  sta RESP0
  sta WSYNC

  ; 25 / 192  -- wait 25
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  sta COLUP0  ; 25
  sta RESP0
  sta WSYNC

  ; 26 / 192  -- wait 26
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  sta RESP0
  sta WSYNC

  ; 27 / 192  -- wait 27
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  sta COLUP0  ; 27
  sta RESP0
  sta WSYNC

  ; 28 / 192  -- wait 28
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  sta RESP0
  sta WSYNC

  ; 29 / 192  -- wait 29
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  sta COLUP0  ; 29
  sta RESP0
  sta WSYNC

  ; 30 / 192  -- wait 30
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  sta RESP0
  sta WSYNC

  ; 31 / 192  -- wait 31
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  sta COLUP0  ; 31
  sta RESP0
  sta WSYNC

  ; 32 / 192  -- wait 32
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  sta RESP0
  sta WSYNC

  ; 33 / 192  -- wait 33
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  sta COLUP0  ; 33
  sta RESP0
  sta WSYNC

  ; 34 / 192  -- wait 34
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  sta RESP0
  sta WSYNC

  ; 35 / 192  -- wait 35
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  sta COLUP0  ; 35
  sta RESP0
  sta WSYNC

  ; 36 / 192  -- wait 36
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  sta RESP0
  sta WSYNC

  ; 37 / 192  -- wait 37
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  sta COLUP0  ; 37
  sta RESP0
  sta WSYNC

  ; 38 / 192  -- wait 38
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  sta RESP0
  sta WSYNC

  ; 39 / 192  -- wait 39
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  sta COLUP0  ; 39
  sta RESP0
  sta WSYNC

  ; 40 / 192  -- wait 40
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  sta RESP0
  sta WSYNC

  ; 41 / 192  -- wait 41
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  sta COLUP0  ; 41
  sta RESP0
  sta WSYNC

  ; 42 / 192  -- wait 42
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  sta RESP0
  sta WSYNC

  ; 43 / 192  -- wait 43
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  sta COLUP0  ; 43
  sta RESP0
  sta WSYNC

  ; 44 / 192  -- wait 44
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  sta RESP0
  sta WSYNC

  ; 45 / 192  -- wait 45
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  sta COLUP0  ; 45
  sta RESP0
  sta WSYNC

  ; 46 / 192  -- wait 46
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  sta RESP0
  sta WSYNC

  ; 47 / 192  -- wait 47
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  sta COLUP0  ; 47
  sta RESP0
  sta WSYNC

  ; 48 / 192  -- wait 48
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  sta RESP0
  sta WSYNC

  ; 49 / 192  -- wait 49
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  sta COLUP0  ; 49
  sta RESP0
  sta WSYNC

  ; 50 / 192  -- wait 50
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  sta RESP0
  sta WSYNC

  ; 51 / 192  -- wait 51
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  sta COLUP0  ; 51
  sta RESP0
  sta WSYNC

  ; 52 / 192  -- wait 52
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  sta RESP0
  sta WSYNC

  ; 53 / 192  -- wait 53
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  sta COLUP0  ; 53
  sta RESP0
  sta WSYNC

  ; 54 / 192  -- wait 54
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  sta RESP0
  sta WSYNC

  ; 55 / 192  -- wait 55
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  sta COLUP0  ; 55
  sta RESP0
  sta WSYNC

  ; 56 / 192  -- wait 56
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  sta RESP0
  sta WSYNC

  ; 57 / 192  -- wait 57
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  sta COLUP0  ; 57
  sta RESP0
  sta WSYNC

  ; 58 / 192  -- wait 58
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  sta RESP0
  sta WSYNC

  ; 59 / 192  -- wait 59
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  sta COLUP0  ; 59
  sta RESP0
  sta WSYNC

  ; 60 / 192  -- wait 60
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  sta RESP0
  sta WSYNC

  ; 61 / 192  -- wait 61
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  sta COLUP0  ; 61
  sta RESP0
  sta WSYNC

  ; 62 / 192  -- wait 62
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  sta RESP0
  sta WSYNC

  ; 63 / 192  -- wait 63
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  sta COLUP0  ; 63
  sta RESP0
  sta WSYNC

  ; 64 / 192  -- wait 64
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  sta RESP0
  sta WSYNC

  ; 65 / 192  -- wait 65
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  sta COLUP0  ; 65
  sta RESP0
  sta WSYNC

  ; 66 / 192  -- wait 66
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  sta RESP0
  sta WSYNC

  ; 67 / 192  -- wait 67
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  sta COLUP0  ; 67
  sta RESP0
  sta WSYNC

  ; 68 / 192  -- wait 68
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  sta RESP0
  sta WSYNC

  ; 69 / 192  -- wait 69
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  sta COLUP0  ; 69
  sta RESP0
  sta WSYNC

  ; 70 / 192  -- wait 70
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  nop         ; 70
  sta RESP0   ; 73
  sta COLUP0  ; 76

  ; 71 / 192  -- wait 71
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  sta COLUP0  ; 71
  sta RESP0   ; 74
  nop         ; 76

  ; 72 / 192  -- wait 72
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  nop         ; 70
  nop         ; 72
  sta RESP0   ; 75

  ; 73 / 192  -- wait 73
  sta COLUP0  ; 2 - single clock holdover from previous scanline
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  nop         ; 70
  sta COLUP0  ; 73
  sta RESP0   ; 76

  ; 74 / 192  - switchover to decrementing wait
  sta COLUBK  ; 3
  lda #0      ; 4
  sta COLUP0
  sta WSYNC

  ; 75 / 192  -- wait 72
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  nop         ; 70
  nop         ; 72
  sta RESP0   ; 75

  ; 76 / 192
  sta WSYNC

  ; 77 / 192  -- wait 71
  sta COLUP0  ; 2 - one clock holdover from previous scanline
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  sta COLUP0  ; 71
  sta RESP0   ; 74
  nop         ; 76

  ; 78 / 192
  sta WSYNC

  ; 79 / 192  -- wait 70
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  nop         ; 70
  sta RESP0   ; 73
  sta COLUP0  ; 76

  ; 80 / 192
  sta WSYNC

  ; 81 / 192  -- wait 69
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  sta COLUP0  ; 69
  sta RESP0
  sta WSYNC

  ; 82 / 192
  sta WSYNC

  ; 83 / 192  -- wait 68
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  nop         ; 68
  sta RESP0
  sta WSYNC

  ; 84 / 192
  sta WSYNC

  ; 85 / 192  -- wait 67
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  sta COLUP0  ; 67
  sta RESP0
  sta WSYNC

  ; 86 / 192
  sta WSYNC

  ; 87 / 192  -- wait 66
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  nop         ; 66
  sta RESP0
  sta WSYNC

  ; 88 / 192
  sta WSYNC

  ; 89 / 192  -- wait 65
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  sta COLUP0  ; 65
  sta RESP0
  sta WSYNC

  ; 90 / 192
  sta WSYNC

  ; 91 / 192  -- wait 64
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  nop         ; 64
  sta RESP0
  sta WSYNC

  ; 92 / 192
  sta WSYNC

  ; 93 / 192  -- wait 63
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  sta COLUP0  ; 63
  sta RESP0
  sta WSYNC

  ; 94 / 192
  sta WSYNC

  ; 95 / 192  -- wait 62
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  nop         ; 62
  sta RESP0
  sta WSYNC

  ; 96 / 192
  sta WSYNC

  ; 97 / 192  -- wait 61
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  sta COLUP0  ; 61
  sta RESP0
  sta WSYNC

  ; 98 / 192
  sta WSYNC

  ; 99 / 192  -- wait 60
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  nop         ; 60
  sta RESP0
  sta WSYNC

  ; 100 / 192
  sta WSYNC

  ; 101 / 192  -- wait 59
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  sta COLUP0  ; 59
  sta RESP0
  sta WSYNC

  ; 102 / 192
  sta WSYNC

  ; 103 / 192  -- wait 58
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  nop         ; 58
  sta RESP0
  sta WSYNC

  ; 104 / 192
  sta WSYNC

  ; 105 / 192  -- wait 57
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  sta COLUP0  ; 57
  sta RESP0
  sta WSYNC

  ; 106 / 192
  sta WSYNC

  ; 107 / 192  -- wait 56
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  nop         ; 56
  sta RESP0
  sta WSYNC

  ; 108 / 192
  sta WSYNC

  ; 109 / 192  -- wait 55
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  sta COLUP0  ; 55
  sta RESP0
  sta WSYNC

  ; 110 / 192
  sta WSYNC

  ; 111 / 192  -- wait 54
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  nop         ; 54
  sta RESP0
  sta WSYNC

  ; 112 / 192
  sta WSYNC

  ; 113 / 192  -- wait 53
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  sta COLUP0  ; 53
  sta RESP0
  sta WSYNC

  ; 114 / 192
  sta WSYNC

  ; 115 / 192  -- wait 52
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  nop         ; 52
  sta RESP0
  sta WSYNC

  ; 116 / 192
  sta WSYNC

  ; 117 / 192  -- wait 51
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  sta COLUP0  ; 51
  sta RESP0
  sta WSYNC

  ; 118 / 192
  sta WSYNC

  ; 119 / 192  -- wait 50
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  nop         ; 50
  sta RESP0
  sta WSYNC

  ; 120 / 192
  sta WSYNC

  ; 121 / 192  -- wait 49
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  sta COLUP0  ; 49
  sta RESP0
  sta WSYNC

  ; 122 / 192
  sta WSYNC

  ; 123 / 192  -- wait 48
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  nop         ; 48
  sta RESP0
  sta WSYNC

  ; 124 / 192
  sta WSYNC

  ; 125 / 192  -- wait 47
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  sta COLUP0  ; 47
  sta RESP0
  sta WSYNC

  ; 126 / 192
  sta WSYNC

  ; 127 / 192  -- wait 46
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  nop         ; 46
  sta RESP0
  sta WSYNC

  ; 128 / 192
  sta WSYNC

  ; 129 / 192  -- wait 45
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  sta COLUP0  ; 45
  sta RESP0
  sta WSYNC

  ; 130 / 192
  sta WSYNC

  ; 131 / 192  -- wait 44
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  nop         ; 44
  sta RESP0
  sta WSYNC

  ; 132 / 192
  sta WSYNC

  ; 133 / 192  -- wait 43
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  sta COLUP0  ; 43
  sta RESP0
  sta WSYNC

  ; 134 / 192
  sta WSYNC

  ; 135 / 192  -- wait 42
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  nop         ; 42
  sta RESP0
  sta WSYNC

  ; 136 / 192
  sta WSYNC

  ; 137 / 192  -- wait 41
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  sta COLUP0  ; 41
  sta RESP0
  sta WSYNC

  ; 138 / 192
  sta WSYNC

  ; 139 / 192  -- wait 40
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  nop         ; 40
  sta RESP0
  sta WSYNC

  ; 140 / 192
  sta WSYNC

  ; 141 / 192  -- wait 39
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  sta COLUP0  ; 39
  sta RESP0
  sta WSYNC

  ; 142 / 192
  sta WSYNC

  ; 143 / 192  -- wait 38
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  nop         ; 38
  sta RESP0
  sta WSYNC

  ; 144 / 192
  sta WSYNC

  ; 145 / 192  -- wait 37
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  sta COLUP0  ; 37
  sta RESP0
  sta WSYNC

  ; 146 / 192
  sta WSYNC

  ; 147 / 192  -- wait 36
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  nop         ; 36
  sta RESP0
  sta WSYNC

  ; 148 / 192
  sta WSYNC

  ; 149 / 192  -- wait 35
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  sta COLUP0  ; 35
  sta RESP0
  sta WSYNC

  ; 150 / 192
  sta WSYNC

  ; 151 / 192  -- wait 34
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  nop         ; 34
  sta RESP0
  sta WSYNC

  ; 152 / 192
  sta WSYNC

  ; 153 / 192  -- wait 33
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  sta COLUP0  ; 33
  sta RESP0
  sta WSYNC

  ; 154 / 192
  sta WSYNC

  ; 155 / 192  -- wait 32
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  nop         ; 32
  sta RESP0
  sta WSYNC

  ; 156 / 192
  sta WSYNC

  ; 157 / 192  -- wait 31
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  sta COLUP0  ; 31
  sta RESP0
  sta WSYNC

  ; 158 / 192
  sta WSYNC

  ; 159 / 192  -- wait 30
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  nop         ; 30
  sta RESP0
  sta WSYNC

  ; 160 / 192
  sta WSYNC

  ; 161 / 192  -- wait 29
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  sta COLUP0  ; 29
  sta RESP0
  sta WSYNC

  ; 162 / 192
  sta WSYNC

  ; 163 / 192  -- wait 28
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  nop         ; 28
  sta RESP0
  sta WSYNC

  ; 164 / 192
  sta WSYNC

  ; 165 / 192  -- wait 27
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  sta COLUP0  ; 27
  sta RESP0
  sta WSYNC

  ; 166 / 192
  sta WSYNC

  ; 167 / 192  -- wait 26
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  nop         ; 26
  sta RESP0
  sta WSYNC

  ; 168 / 192
  sta WSYNC

  ; 169 / 192  -- wait 25
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  sta COLUP0  ; 25
  sta RESP0
  sta WSYNC

  ; 170 / 192
  sta WSYNC

  ; 171 / 192  -- wait 24
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  nop         ; 24
  sta RESP0
  sta WSYNC

  ; 172 / 192
  sta WSYNC

  ; 173 / 192  -- wait 23
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  sta COLUP0  ; 23
  sta RESP0
  sta WSYNC

  ; 174 / 192
  sta WSYNC

  ; 175 / 192  -- wait 22
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  nop         ; 22
  sta RESP0
  sta WSYNC

  ; 176 / 192
  sta WSYNC

  ; 177 / 192  -- wait 21
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  sta COLUP0  ; 21
  sta RESP0
  sta WSYNC

  ; 178 / 192
  sta WSYNC

  ; 179 / 192  -- wait 20
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  nop         ; 20
  sta RESP0
  sta WSYNC

  ; 180 / 192
  sta WSYNC

  ; 181 / 192  -- wait 19
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  sta COLUP0  ; 19
  sta RESP0
  sta WSYNC

  ; 182 / 192
  sta WSYNC

  ; 183 / 192  -- wait 18
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  nop         ; 18
  sta RESP0
  sta WSYNC

  ; 184 / 192
  sta WSYNC

  ; 185 / 192  -- wait 17
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  sta COLUP0  ; 17
  sta RESP0
  sta WSYNC

  ; 186 / 192
  sta WSYNC

  ; 187 / 192  -- wait 16
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  nop         ; 16
  sta RESP0
  sta WSYNC

  ; 188 / 192
  sta WSYNC

  ; 189 / 192  -- wait 15
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  sta COLUP0  ; 15
  sta RESP0
  sta WSYNC

  ; 190 / 192
  sta WSYNC

  ; 191 / 192  -- wait 14
  nop         ; 2
  nop         ; 4
  nop         ; 6
  nop         ; 8
  nop         ; 10
  nop         ; 12
  nop         ; 14
  sta COLUP0
  sta RESP0
  sta WSYNC

  ; 192 / 192
  sta WSYNC

  ; turn on vblank, then 30 scanlines of overscan
  lda #$42
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

  jmp StartOfFrame

  ORG $FFFA
  .word Reset
  .word Reset
  .word Reset
END
