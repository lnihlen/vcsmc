#ifndef SRC_OPCODE_H_
#define SRC_OPCODE_H_

#include "types.h"
#include "constants.h"

namespace vcsmc {

inline uint32 OpCodeBytes(OpCode op) {
  switch (op) {
    case JMP_Absolute:
      return 3;

    case LDA_Immediate:
    case LDX_Immediate:
    case LDY_Immediate:
      return 2;

    case NOP_Implied:
      return 1;

    case STA_ZeroPage:
    case STX_ZeroPage:
    case STY_ZeroPage:
      return 2;
  }
  return 0;
}

inline uint32 OpCodeCycles(OpCode op) {
  switch (op) {
    case JMP_Absolute:
      return 3;

    case LDA_Immediate:
    case LDX_Immediate:
    case LDY_Immediate:
      return 2;

    case NOP_Implied:
      return 2;

    case STA_ZeroPage:
    case STX_ZeroPage:
    case STY_ZeroPage:
      return 3;
  }
  return 0;
}

}  // namespace vcsmc

#endif  // SRC_OPCODE_H_
