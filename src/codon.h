#ifndef SRC_CODON_H_
#define SRC_CODON_H_

#include <cassert>

#include "constants.h"
#include "types.h"

namespace vcsmc {

// A Codon is a packed uint32 containing, from least to most significant byte:
//  * A Codon Action, usually setting all or part of TIA register
//  * An action parameter, if TIA register it's the address of the register
//  * The target byte value for the TIA register, and
//  * A don't care mask, where bit value 0 means the same bit in the target
//    byte value is a don't care.

enum Action : uint8 {
  kSetPF0          = 0,
  kSetPF1          = 1,
  kSetPF2          = 2,
  kSetCTRLPF_REF   = 3,
  kSetCTRLPF_SCORE = 4,
  kSetCTRLPF_PFP   = 5,
  kSetCTRLPF_BALL  = 6,
  kSetNUSIZ0_P0    = 7,
  kSetNUSIZ0_M0    = 8,
  kSetNUSIZ1_P1    = 9,
  kSetNUSIZ1_M1    = 10,
  kStrobeRESP0     = 11,
  kStrobeRESP1     = 12,
  kStrobeRESM0     = 13,
  kStrobeRESM1     = 14,
  kStrobeRESBL     = 15,
  kSetRESMP0       = 16,
  kSetRESMP1       = 17,
  kSetENAM0        = 18,
  kSetENAM1        = 19,
  kSetENABL        = 20,
  kSetGRP0         = 21,
  kSetGRP1         = 22,
  kSetREFP0        = 23,
  kSetREFP1        = 24,
  kSetVDELP0       = 25,
  kSetVDELP1       = 26,
  kSetVDELBL       = 27,
  kSetCOLUP0       = 28,
  kSetCOLUP1       = 29,
  kSetCOLUPF       = 30,
  kSetCOLUBK       = 31,
  kWait            = 32,

  // The following Codons aren't included in the generated table but can be
  // Translated into bytecode, as they are useful for frame programming, bank
  // generation, or audio modulation.
  kSwitchBanks     = 33
};

// We compute the theoretical upper limit of load/store instructions the 6502
// could execute within the timing of a single frame. The fastest opcodes
// consume 2 CPU cycles, and any additional frame requirements like audio or
// vertical blanking will only add operations during transcription, thus
// increasing frame time, so a reasonable upper bound of codons per frame can
// be computed as the number of 2-cycle instructions the CPU can execute in
// one frame.
const uint32 kFrameSizeCodons = kScreenSizeCycles / 2;

typedef uint32 Codon;

// Given an action, parameter, tia_value, and mask, returns a packed uint32
// with those values.
inline Codon PackCodon(Action action,
                       uint8 action_parameter,
                       uint8 tia_value,
                       uint8 tia_mask) {
  return static_cast<uint32>(action) |
         static_cast<uint32>(action_parameter) << 8 |
         static_cast<uint32>(tia_value) << 16 |
         static_cast<uint32>(tia_mask) << 24;
}

inline Action CodonAction(Codon codon) {
  return static_cast<Action>(codon & 0x000000ff);
}

inline uint8 CodonActionParameter(Codon codon) {
  return static_cast<uint8>((codon >> 8) & 0x000000ff);
}

inline uint8 CodonTIAValue(Codon codon) {
  return static_cast<uint8>((codon >> 16) & 0x000000ff);
}

inline uint8 CodonTIAMask(Codon codon) {
  return static_cast<uint8>((codon >> 24) & 0x000000ff);
}

inline Codon MakeWaitCodon(uint8 duration) {
  return PackCodon(kWait, duration, 0x00, 0xff);
}

inline Codon MakeTIACodon(Action action, uint8 value) {
  TIA tia;
  uint8 mask;
  switch (action) {
    case kSetPF0:
      tia = PF0;
      mask = 0xf0;
      break;

    case kSetPF1:
      tia = PF1;
      mask = 0xff;
      break;

    case kSetPF2:
      tia = PF2;
      mask = 0xff;
      break;

    case kSetCTRLPF_REF:
      tia = CTRLPF;
      mask = 0x01;
      break;

    case kSetCTRLPF_SCORE:
      tia = CTRLPF;
      mask = 0x02;
      break;

    case kSetCTRLPF_PFP:
      tia = CTRLPF;
      mask = 0x04;
      break;

    case kSetCTRLPF_BALL:
      tia = CTRLPF;
      mask = 0x30;
      break;

    case kSetNUSIZ0_P0:
      tia = NUSIZ0;
      mask = 0x07;
      break;

    case kSetNUSIZ0_M0:
      tia = NUSIZ0;
      mask = 0x30;
      break;

    case kSetNUSIZ1_P1:
      tia = NUSIZ1;
      mask = 0x07;
      break;

    case kSetNUSIZ1_M1:
      tia = NUSIZ1;
      mask = 0x30;
      break;

    case kStrobeRESP0:
      tia = RESP0;
      mask = 0x00;
      break;

    case kStrobeRESP1:
      tia = RESP1;
      mask = 0x00;
      break;

    case kStrobeRESM0:
      tia = RESM0;
      mask = 0x00;
      break;

    case kStrobeRESM1:
      tia = RESM1;
      mask = 0x00;
      break;

    case kStrobeRESBL:
      tia = RESBL;
      mask = 0x00;
      break;

    case kSetRESMP0:
      tia = RESMP0;
      mask = 0x02;
      break;

    case kSetRESMP1:
      tia = RESMP1;
      mask = 0x02;
      break;

    case kSetENAM0:
      tia = ENAM0;
      mask = 0x02;
      break;

    case kSetENAM1:
      tia = ENAM1;
      mask = 0x02;
      break;

    case kSetENABL:
      tia = ENABL;
      mask = 0x02;
      break;

    case kSetGRP0:
      tia = GRP0;
      mask = 0xff;
      break;

    case kSetGRP1:
      tia = GRP1;
      mask = 0xff;
      break;

    case kSetREFP0:
      tia = REFP0;
      mask = 0x08;
      break;

    case kSetREFP1:
      tia = REFP1;
      mask = 0x08;
      break;

    case kSetVDELP0:
      tia = VDELP0;
      mask = 0x01;
      break;

    case kSetVDELP1:
      tia = VDELP1;
      mask = 0x01;
      break;

    case kSetVDELBL:
      tia = VDELBL;
      mask = 0x01;
      break;

    case kSetCOLUP0:
      tia = COLUP0;
      mask = 0xfe;
      break;

    case kSetCOLUP1:
      tia = COLUP1;
      mask = 0xfe;
      break;

    case kSetCOLUPF:
      tia = COLUPF;
      mask = 0xfe;
      break;

    case kSetCOLUBK:
      tia = COLUBK;
      mask = 0xfe;
      break;

    default:
      assert(false);
      break;
  }

  return PackCodon(action, tia, value, mask);
}

// Given a number of padding bytes construct a bank switch Codon with the
// supplied amount of padding to round out the bank size to kBankSize.
inline Codon MakeBankSwitchCodon(uint8 padding) {
  return PackCodon(kSwitchBanks, padding, 0, 0);
}

}  // namespace vcsmtc

#endif  // SRC_CODON_H_
