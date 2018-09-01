#ifndef SRC_CODON_H_
#define SRC_CODON_H_

#include "types.h"

namespace vcsmc {

enum Codon : uint32 {
  kSetPF0 = 0,                            // 4 bits
  kSetPF1 = kSetPF0 + 16,                 // 8 bits
  kSetPF2 = kSetPF1 + 256,                // 8 bits
  kSetCTRLPF_REF = kSetPF2 + 256,         // 1 bit
  kSetCTRLPF_SCORE = kSetCTRLPF_REF + 2,  // 1 bit
  kSetCTRLPF_PFP = kSetCTRLPF_SCORE + 2,  // 1 bit
  kSetCTRLPF_BALL = kSetCTRLPF_PFP + 2,   // 2 bits
  kSetNUSIZ0_P0 = kSetCTRLPF_BALL + 4,    // 3 bits
  kSetNUSIZ0_M0 = kSetNUSIZ0_P0 + 8,      // 2 bits
  kSetNUSIZ1_P1 = kSetNUSIZ0_M0 + 4,      // 3 bits
  kSetNUSIZ1_M1 = kSetNUSIZ1_P1 + 8,      // 2 bits
  kSetRESP0 = kSetNUSIZ1_M1 + 4,          // no bits (strobe)
  kSetRESP1 = kSetRESP0 + 1,              // no bits (strobe)
  kSetRESM0 = kSetRESP1 + 1,              // no bits (strobe)
  kSetRESM1 = kSetRESM0 + 1,              // no bits (strobe)
  kSetRESBL = kSetRESM1 + 1,              // no bits (strobe)
  kSetRESMP0 = kSetRESBL + 1,             // 1 bit
  kSetRESMP1 = kSetRESMP0 + 2,            // 1 bit
  kSetENAM0 = kSetRESMP1 + 2,             // 1 bit
  kSetENAM1 = kSetENAM0 + 2,              // 1 bit
  kSetENABL = kSetENAM1 + 2,              // 1 bit
  kSetGRP0 = kSetENABL + 2,               // 8 bits
  kSetGRP1 = kSetGRP0 + 256,              // 8 bits
  kSetREFP0 = kSetGRP1 + 256,             // 1 bit
  kSetREFP1 = kSetREFP0 + 2,              // 1 bit
  kSetVDELP0 = kSetREFP1 + 2,             // 1 bit
  kSetVDELP1 = kSetVDELP0 + 2,            // 1 bit
  kSetCOLUP0 = kSetVDELP1 + 2,            // 7 bits
  kSetCOLUP1 = kSetCOLUP0 + 127,          // 7 bits
  kSetCOLUPF = kSetCOLUP1 + 127,          // 7 bits
  kSetCOLUBK = kSetCOLUPF + 127,          // 7 bits
  kCodonCount = kSetCOLUBK + 127,
  kWait = kCodonCount
};

}  // namespace vcsmtc

#endif  // SRC_CODON_H_
