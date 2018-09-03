#ifndef SRC_CODON_H_
#define SRC_CODON_H_

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
  kWait            = 32
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
inline Codon MakeCodon(Action action,
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

}  // namespace vcsmtc

#endif  // SRC_CODON_H_
