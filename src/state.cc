#include "state.h"

#include <cassert>
#include <cstring>
#include <iostream>

#include "color.h"
#include "colu_strip.h"
#include "spec.h"

namespace vcsmc {

State::State()
    : tia_known_(0),
      registers_known(0),
      range_(0, kFrameSizeClocks) {
  std::memset(tia_, 0, sizeof(tia_));
  std::memset(registers_, 0, sizeof(registers_));
}

void State::PaintInto(ColuStrip* colu_strip) {
  uint32 local_clock = range_.start_time() % kScanLineWidthClocks;
  uint32 local_until = local_clock + range_.duration();
  uint32 starting_clock = std::max(local_clock, kHBlankWidthClocks);
  uint32 starting_column = starting_clock - kHBlankWidthClocks;
  for (uint32 clock = starting_clock; clock < local_until; ++clock) {
    uint8 colu = tia_[TIA::COLUBK];
    if (PlayfieldPaints(clock)) {
      colu = tia_[TIA::COLUPF];
    }

    colu_strip->set_colu(starting_column++, colu);
  }
}

const uint32 State::EarliestTimeAfter(const Spec& spec) const {
  Range within(IntersectRanges(range_, spec.range()));
  if (within.IsEmpty())
    return kInfinity;

  switch(spec.tia()) {
    // Not supported currently.
    case TIA::VSYNC:
    case TIA::VBLANK:
    case TIA::WSYNC:
    case TIA::RSYNC:
      return kInfinity;

    // NUSIZ support coming later.
    case TIA::NUSIZ0:
    case TIA::NUSIZ1:
      return kInfinity;

    case TIA::COLUP0:
      return EarliestPlayerPaints(false, within);

    case TIA::COLUP1:
      return EarliestPlayerPaints(true, within);

    case TIA::COLUPF:
      return EarliestPlayfieldPaints(within);

    case TIA::COLUBK:
      return EarliestBackgroundPaints(within);

    // Not currently supported.
    case TIA::CTRLPF:
      return kInfinity;

    case REFP0:
      // anywhere player _could_ paint
  }
}

std::unique_ptr<State> State::Clone() const {
  std::unique_ptr<State> state(new State(*this));
  return state;
}

std::unique_ptr<State> State::AdvanceTime(uint32 delta) {
  std::unique_ptr<State> state(Clone());
  uint32 new_start_time = range_.start_time() + delta;
  range_.set_end_time(new_start_time);
  state->range_.set_start_time(new_start_time);
  return state;
}

std::unique_ptr<State> State::AdvanceTimeAndSetRegister(
    uint32 delta, Register reg, uint8 value) {
  std::unique_ptr<State> state(AdvanceTime(delta));
  state->registers_known_ |= (1 << static_cast<int>(reg));
  state->registers_[reg] = value;
  return state;
}

std::unique_ptr<State> State::AdvanceTimeAndCopyRegisterToTIA(
    uint32 delta, Register reg, TIA address) const {
  std::unique_ptr<State> state(AdvanceTime(delta));
  state->tia_known_ |= (1 << static_cast<int>(address));
  uint8 reg_value = state->registers_[reg];
  switch (address) {
    // interesting code for strobes and the like here... :)

    default:
      state->tia_[address] = reg_value;
      break;
  }
  return state;
}

// static
std::string State::RegisterToString(const Register reg) {
  if (reg == Register::A) {
    return "a";
  } else if (reg == Register::X) {
    return "x";
  }

  return "y";
}

// static
std::string State::AddressToString(const uint8 address) {
  // First see if within the TIA range or no:
  if (address < TIA::TIA_COUNT) {
    switch (address) {
      case VSYNC:   return "VSYNC";
      case VBLANK:  return "VBLANK";
      case WSYNC:   return "WSYNC";
      case RSYNC:   return "RSYNC";
      case NUSIZ0:  return "NUSIZ0";
      case NUSIZ1:  return "NUSIZ1";
      case COLUP0:  return "COLUP0";
      case COLUP1:  return "COLUP1";
      case COLUPF:  return "COLUPF";
      case COLUBK:  return "COLUBK";
      case CTRLPF:  return "CTRLPF";
      case REFP0:   return "REFP0";
      case REFP1:   return "REFP1";
      case PF0:     return "PF0";
      case PF1:     return "PF1";
      case PF2:     return "PF2";
      case RESP0:   return "RESP0";
      case RESP1:   return "RESP1";
      case RESM0:   return "RESM0";
      case RESM1:   return "RESM1";
      case RESBL:   return "RESBL";
      case AUDC0:   return "AUDC0";
      case AUDC1:   return "AUDC1";
      case AUDF0:   return "AUDF0";
      case AUDF1:   return "AUDF1";
      case AUDV0:   return "AUDV0";
      case AUDV1:   return "AUDV1";
      case GRP0:    return "GRP0";
      case GRP1:    return "GRP1";
      case ENAM0:   return "ENAM0";
      case ENAM1:   return "ENAM1";
      case ENABL:   return "ENABL";
      case HMP0:    return "HMP0";
      case HMP1:    return "HMP1";
      case HMM0:    return "HMM0";
      case HMM1:    return "HMM1";
      case HMBL:    return "HMBL";
      case VDELP0:  return "VDELP0";
      case VDELP1:  return "VDELP1";
      case RESMP0:  return "RESMP0";
      case RESMP1:  return "RESMP1";
      case HMOVE:   return "HMOVE";
      case HMCLR:   return "HMCLR";
      case CXCLR:   return "CXCLR";
      default:      return "???";
    }
  }
  // Must be a memory address, just print as one.
  return ByteToHexString(address);
}

// static
std::string State::ByteToHexString(const uint8 value) {
  // Print the number value in hex to a stack buffer and wrap into string.
  char buf[4];
  std::snprintf(buf, 4, "$%01x", value);
  return std::string(buf);
}

State::State(const State& state) {
  std::memcpy(tia_, state.tia_, TIA_COUNT);
  std::memcpy(registers_, state.registers_, REGISTER_COUNT);
  tia_known_ = state.tia_known_;
  registers_known_ = state.registers_known_;
  range_ = state.range_;
}

const uint32 State::EarliestPlayerPaints(bool p1, const Range& within) const {
  // If player graphics are zero then they will never paint.
  TIA grp = p1 ? TIA::GRP1 : TIA::GRP0;
  if (tia_[grp] == 0)
    return 0;

  // Start from the end of our |range_| and work our way back in time, stopping
  // whenever we encounter a pixel that is actually painted by the player. If
  // no such pixel is encountered then we return 0 to indicate that the spec
  // can be scheduled before our |range_| time.
  uint32 duration = within.Duration();
  assert(within.end_time() >= duration);
  for (uint32 i = 0; i < duration; ++i) {
    uint32 color_clock = within.end_time() - i - 1;
    uint32 local_clock = color_clock % kScanLineWidthClocks;
    if (PlayerPaints(p1, local_clock))
      return color_clock;
  }
  return 0;
}

const bool State::PlayerPaints(bool p1, uint32 local_clock) const {
  // TODO: write me
  return false;
}

const bool State::GRP0CouldPaint(uint32 local_clock) const {
  // TODO: write me
  return false;
}

const bool State::GRP1CouldPaint(uint32 local_clock) const {
  // TODO: write me
  return false;
}

const uint32 State::EarliestPlayfieldPaints(const Range& within) const {
  uint32 duration = within.Duration();
  assert(within.end_time() >= duration);
  for (uint32 i = 0; i < duration; ++i) {
    uint32 color_clock = within.end_time() - i - 1;
    uint32 local_clock = color_clock % kScanLineWidthClocks;
    if (PlayfieldPaints(local_clock))
      return color_clock;
  }
  return 0;
}

const bool State::PlayfieldPaints(uint32 local_clock) {
  assert(clock >= 68);

  if (PF0CouldPaint(local_clock)) {
    // PF0 D4 through D7 left to right.
    int pfbit = (local_clock - (local_clock < 84 ? 68 : 148)) >> 2;
    return tia_[TIA::PF0] & (0x10 << pfbit);
  } else if (PF1CouldPaint(local_clock)) {
    // PF1 D7 through D0 left to right.
    int pfbit = (locl_clock - (local_clock < 116 ? 84 : 164)) >> 2;
    return tia_[TIA::PF1] & (0x80 >> pfbit);
  } else {
    // PF2 D0 through D7 left to right.
    assert(PF2CouldPaint(local_clock));
    int pfbit = (local_clock - (local_clock < 148 ? 116 : 196)) >> 2;
    return tia_[TIA::PF2] & (0x01 << pfbit);
  }
}

const bool State::PF0CouldPaint(uint32 local_clock) const {
  return local_clock < 84 || (local_clock >= 148 && local_clock < 164);
}

const bool State::PF1CouldPaint(uint32 local_clock) const {
  return local_clock < 116 || (local_clock >= 164 && local_clock < 196);
}

const bool State::PF2CouldPaint(uint32 local_clock) const {
  return local_clock < 148 || (local_clock >= 196 && local_clock < 228);
}

const bool State::EarliestBackgroundPaints(const Range& within) const {
  uint32 duration = within.Duration();
  assert(within.end_time() >= duration);
  for (uint32 i = 0; i < duration; ++i) {
    uint32 color_clock = within.end_time() - i - 1;
    uint32 local_clock = color_clock % kScanLineWidthClocks;
    if (local_clock < kHBlankWidthClocks)
      continue;
    if (PlayerPaints(false, local_clock))
      continue;
    if (PlayerPaints(true, local_clock))
      continue;
    if (PlayfiedPaints(local_clock))
      continue;
    // Neither player nor playfield paints, must be the bg color that paints.
    return color_clock;
  }
  return 0;
}

}  // namespace vcsmc
