#include "state.h"

#include <cassert>

namespace vcsmc {

State::State(uint32 current_time) : current_time_(current_time) {
  tia_.fill(0);
  registers_.fill(0);
  register_last_used_.fill(0);
}

Snippet State::Sequence(Codon codon) const {
  // Unpack Codon argument.
  Action action = CodonAction(codon);
  uint8 parameter = CodonActionParameter(codon);
  uint8 tia_value = CodonTIAValue(codon);
  uint8 tia_mask = CodonTIAMask(codon);
  Snippet snippet;

  if (action == kWait) {
    int wait_time = parameter;
    while (wait_time > 3) {
      snippet.Insert(NOP_Implied);
      wait_time -= 2;
    }

    if (wait_time == 3) {
      // We use a BIT test on the VSYNC register, which changes no meaningful
      // state but flags on the CPU, and consumes 3 cycles of time in the
      // process.
      snippet.Insert(BIT_ZeroPage);
      snippet.Insert(VSYNC);
    } else {
      snippet.Insert(NOP_Implied);
    }

    snippet.duration = parameter;
    snippet.should_advance_register_rotation = false;
    return snippet;
  }

  // From this point forward we assume that the CodonAction is some state
  // change on the TIA.
  assert(action < kWait);

  // Codons for CTRLPF and NUSIZ pack two different state changes into one
  // register. We incorporate the value of the register from the current state
  // for the parts we aren't changing into the mask, to keep the state changes
  // hermetic. We also check the current value of tia register against the
  // target value (within mask) to see if any work at all needs to be done.
  uint8 current_tia = parameter < TIA_COUNT ? tia_[parameter] : 0;

  // Strobes don't care about register values, only timing of the write, and
  // so can use any register for bit. However, they can't be optimized out.
  bool is_strobe = (action == kStrobeRESP0) ||
                   (action == kStrobeRESP1) ||
                   (action == kStrobeRESM0) ||
                   (action == kStrobeRESM1) ||
                   (action == kStrobeRESBL);

  // It could be that the current value is already set in the TIA, making this
  // Codon a no-op, in which case we return the current empty Snippet.
  if (!is_strobe && ((tia_value & tia_mask) == (current_tia & tia_mask))) {
    return snippet;
  }

  switch (action) {
    case kSetCTRLPF_REF:
      // Preserve current state of CTRLPF bits 1,2,4,5.
      tia_value = (tia_value & 0x01) | (current_tia & 0b00110110);
      tia_mask = 0x37;
      break;

    case kSetCTRLPF_SCORE:
      // Preserve current state of CTRLPF bits 0,2,4,5.
      tia_value = (tia_value & 0x02) | (current_tia & 0b00110101);
      tia_mask = 0x37;
      break;

    case kSetCTRLPF_PFP:
      // Preserve current state of CTRLF bits 0,1,4,5.
      tia_value = (tia_value & 0x04) | (current_tia & 0b00110011);
      tia_mask = 0x37;
      break;

    case kSetCTRLPF_BALL:
      // Preserve current state of CTRLPF bits 0,1,2.
      tia_value = (tia_value & 0x30) | (current_tia & 0b00000111);
      tia_mask = 0x37;
      break;

    case kSetNUSIZ0_P0:
    case kSetNUSIZ1_P1:
      // Preserve current state of NUSIZ{0,1} bits 4,5.
      tia_value = (tia_value & 0x07) | (current_tia & 0b00110000);
      tia_mask = 0x37;
    break;

    case kSetNUSIZ0_M0:
    case kSetNUSIZ1_M1:
      // Preserve current state of NUSIZ{0,1} bits 0,1,2.
      tia_value = (tia_value & 0x30) | (current_tia & 0b00000111);
      tia_mask = 0x37;
      break;

    default:
      break;
  }

  // Check registers for possible match that fits needed value within mask.
  Register store_register;

  if (is_strobe || ((registers_[A] & tia_mask) == tia_value)) {
    store_register = A;
  } else if ((registers_[X] & tia_mask) == tia_value) {
    store_register = X;
  } else if ((registers_[Y] & tia_mask) == tia_value) {
    store_register = Y;
  } else {
    // No matching re-usable register found, we will need to issue a load
    // instruction to load the needed value into a register. We pick the least
    // recently used register in the hopes of increasing register value re-use.
    uint32 oldest_use_time = register_last_used_[A];
    store_register = A;
    if (register_last_used_[X] < oldest_use_time) {
      oldest_use_time = register_last_used_[X];
      store_register = X;
    }
    if (register_last_used_[Y] < oldest_use_time) {
      oldest_use_time = register_last_used_[Y];
      store_register = Y;
    }

    // We have selected a load target for the register, issue the load command.
    switch (store_register) {
      case A:
        snippet.Insert(LDA_Immediate);
        break;

      case X:
        snippet.Insert(LDX_Immediate);
        break;

      case Y:
        snippet.Insert(LDY_Immediate);
        break;

      default:
        assert(false);
        break;
    }

    snippet.Insert(tia_value);
    snippet.duration += 2;
  }

  assert(store_register != REGISTER_COUNT);

  switch (store_register) {
    case A:
      snippet.Insert(STA_ZeroPage);
      break;

    case X:
      snippet.Insert(STX_ZeroPage);
      break;

    case Y:
      snippet.Insert(STY_ZeroPage);
      break;

    default:
      assert(false);
      break;
  }

  snippet.Insert(parameter);
  snippet.duration += 3;

  // Normal stores should update register usage, but strobes should not.
  snippet.should_advance_register_rotation = !is_strobe;

  return snippet;
}

void State::Apply(const Snippet& snippet) {
  size_t offset = 0;
  uint32 starting_time = current_time_;

  while (offset < snippet.size) {
    OpCode op = static_cast<OpCode>(snippet.bytecode[offset]);
    ++offset;

    switch (op) {
      case BIT_ZeroPage:
        ++offset;
        current_time_ += 3;
        break;

      case JMP_Absolute:
        offset += 2;
        current_time_ += 3;
        break;

      case LDA_Immediate:
        registers_[A] = snippet.bytecode[offset];
        ++offset;
        current_time_ += 2;
        break;

      case LDX_Immediate:
        registers_[X] = snippet.bytecode[offset];
        ++offset;
        current_time_ += 2;
        break;

      case LDY_Immediate:
        registers_[Y] = snippet.bytecode[offset];
        ++offset;
        current_time_ += 2;
        break;

      case NOP_Implied:
        current_time_ += 2;
        break;

      case STA_ZeroPage:
        tia_[snippet.bytecode[offset]] = registers_[A];
        ++offset;
        current_time_ += 3;
        if (snippet.should_advance_register_rotation) {
          register_last_used_[A] = current_time_;
        }
        break;

      case STX_ZeroPage:
        tia_[snippet.bytecode[offset]] = registers_[X];
        ++offset;
        current_time_ += 3;
        if (snippet.should_advance_register_rotation) {
          register_last_used_[X] = current_time_;
        }
        break;

      case STY_ZeroPage:
        tia_[snippet.bytecode[offset]] = registers_[Y];
        current_time_ += 3;
        if (snippet.should_advance_register_rotation) {
          register_last_used_[Y] = current_time_;
        }
        break;

      default:
        assert(false);
        break;
    }
  }

  assert(starting_time + snippet.duration == current_time_);
}

}  // namespace vcsmc
