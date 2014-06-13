#include "state.h"

#include <cstring>
#include <iostream>

#include "color.h"
#include "colu_strip.h"
#include "spec.h"

namespace vcsmc {

State::State()
    : tia_known_(kTIAStrobeMask),
      registers_known_(0),
      range_(0, kFrameSizeClocks) {
  // The unknown/known logic asserts on |tia_| or |registers_| access, so we
  // can skip initialization of their memory areas.
}

std::unique_ptr<State> State::Clone() const {
  std::unique_ptr<State> state(new State(*this));
  return state;
}

std::unique_ptr<State> State::MakeEntryState() const {
  std::unique_ptr<State> state(Clone());
  state->registers_known_ = 0;
  state->range_ = Range(range_.end_time(), range_.end_time());
  return state;
}

std::unique_ptr<State> State::AdvanceTime(uint32 delta) {
  assert(delta > 0);
  std::unique_ptr<State> state(Clone());
  uint32 new_start_time = range_.start_time() + delta;
  range_.set_end_time(new_start_time);
  state->range_.set_start_time(new_start_time);
  return state;
}

std::unique_ptr<State> State::AdvanceTimeAndSetRegister(
    uint32 delta, Register axy, uint8 value) {
  std::unique_ptr<State> state(AdvanceTime(delta));
  state->registers_known_ |= (1 << static_cast<int>(axy));
  state->registers_[axy] = value;
  return state;
}

std::unique_ptr<State> State::AdvanceTimeAndCopyRegisterToTIA(
    uint32 delta, Register axy, TIA address) {
  std::unique_ptr<State> state(AdvanceTime(delta));
  uint8 reg_value = state->reg(axy);
  switch (address) {
    // interesting code for strobes and the like here... :)

    // TODO: consider adding an assert here to verify that EarliestTimeAfter()
    // would actually allow the transition during the duration of |this| state.

    default:
      state->tia_known_ |= (1 << static_cast<int>(address));
      state->tia_[address] = reg_value;
      break;
  }
  return state;
}

void State::PaintInto(ColuStrip* colu_strip) const {
  Range strip_range(Range::IntersectRanges(colu_strip->range(), range_));
  uint32 local_clock = strip_range.start_time() % kScanLineWidthClocks;
  uint32 local_until = local_clock + strip_range.Duration();
  uint32 starting_clock = std::max(local_clock, kHBlankWidthClocks);
  uint32 starting_column = starting_clock - kHBlankWidthClocks;
  for (uint32 clock = starting_clock; clock < local_until; ++clock) {
    uint8 colu = tia(TIA::COLUBK);
    if (PlayfieldPaints(clock)) {
      // If D1 of CTRLPF is set the playfield paints with COLUP0 on the left
      // side and COLUP1 on the right side.
      if (tia(TIA::CTRLPF) & 0x02)
        colu = clock < 148 ? tia(TIA::COLUP0) : tia(TIA::COLUP1);
      else
        colu = tia(TIA::COLUPF);
    }

    colu_strip->set_colu(starting_column++, colu);
  }
}

const uint32 State::EarliestTimeAfter(const Spec& spec) const {
  Range within(Range::IntersectRanges(range_, spec.range()));
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
      // If in score mode we ignore COLUPF for all playfield rendering.
      if (tia(TIA::CTRLPF) & 0x02)
        return 0;
      return EarliestPlayfieldPaints(within);

    case TIA::COLUBK:
      return EarliestBackgroundPaints(within);

    // Not currently supported.
    case TIA::CTRLPF:
      return kInfinity;

    case TIA::REFP0:
      return EarliestPlayerCouldPaint(false, within);

    case TIA::REFP1:
      return EarliestPlayerCouldPaint(true, within);

    case TIA::PF0:
      return EarliestPF0CouldPaint(within);

    case TIA::PF1:
      return EarliestPF1CouldPaint(within);

    case TIA::PF2:
      return EarliestPF2CouldPaint(within);

    // These are strobe registers and need to happen right when they need to
    // happen, so are not subject to back-scheduling. Some thought to be given
    // about how to handle timing on these.
    case TIA::RESP0:
    case TIA::RESP1:
    case TIA::RESM0:
    case TIA::RESM1:
    case TIA::RESBL:
      return kInfinity;

    // Audio registers updated at fixed intervals and not supported by
    // graphics scheduler.
    case TIA::AUDC0:
    case TIA::AUDC1:
    case TIA::AUDF0:
    case TIA::AUDF1:
    case TIA::AUDV0:
    case TIA::AUDV1:
      return kInfinity;

    case TIA::GRP0:
      return EarliestPlayerCouldPaint(false, within);

    case TIA::GRP1:
      return EarliestPlayerCouldPaint(true, within);

    // Ball and missile support coming later.
    case TIA::ENAM0:
    case TIA::ENAM1:
    case TIA::ENABL:
      return kInfinity;

    // Some tricky timing considerations here. Leave alone for now but when
    // adding player supprt this will need revision. These can be written any
    // time after a prior strobe of HMOVE.
    case TIA::HMP0:
    case TIA::HMP1:
      return kInfinity;

    // Pending missile and ball support.
    case TIA::HMM0:
    case TIA::HMM1:
    case TIA::HMBL:
      return kInfinity;

    // While potentially quite powerful we will ignore these for now.
    case TIA::VDELP0:
    case TIA::VDELP1:
      return kInfinity;

    // Pending missile and ball support.
    case TIA::VDELBL:
    case TIA::RESMP0:
    case TIA::RESMP1:
      return kInfinity;

    // Another strobe with zero scheduling flexibility.
    case TIA::HMOVE:
      return kInfinity;

    // Potential conflicts with setting any of the other horizontal motion
    // registers, but otherwise represents an efficient way to get back to a
    // known state.
    case TIA::HMCLR:
      return kInfinity;

    // Non-interactive software has no use for a collision latch that I can
    // think of.
    case TIA::CXCLR:
      return kInfinity;

    case TIA::TIA_COUNT:
      return kInfinity;
  }

  return kInfinity;
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
  if (tia(grp) == 0)
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

const uint32 State::EarliestPlayerCouldPaint(
    bool p1, const Range& within) const {
  uint32 duration = within.Duration();
  assert(within.end_time() >= duration);
  for (uint32 i = 0; i < duration; ++i) {
    uint32 color_clock = within.end_time() - i - 1;
    uint32 local_clock = color_clock % kScanLineWidthClocks;
    if (PlayerCouldPaint(p1, local_clock))
      return color_clock;
  }
  return 0;
}

const bool State::PlayerPaints(bool p1, uint32 local_clock) const {
  // TODO: write me
  return false;
}

const bool State::PlayerCouldPaint(bool p1, uint32 local_clock) const {
  // TODO: write me
  return false;
}

// playfield schedules
// +-----------+------+---------------------+--------------------+
// | playfield | left | right mirroring off | right mirroring on |
// +-----------+------+---------------------+--------------------+
// |    PF0    |  68  |        148          |        212         |
// |    PF1    |  84  |        164          |        180         |
// |    PF2    | 116  |        196          |        148         |
// +-----------+------+---------------------+--------------------+

const uint32 State::EarliestPlayfieldPaints(const Range& within) const {
  uint32 duration = within.Duration();
  assert(within.end_time() >= duration);
  for (uint32 i = 1; i <= duration; ++i) {
    uint32 color_clock = within.end_time() - i;
    uint32 local_clock = color_clock % kScanLineWidthClocks;
    if (local_clock < kHBlankWidthClocks)
      continue;

    if (PlayfieldPaints(local_clock))
      return color_clock;
  }
  return 0;
}

const uint32 State::EarliestPF0CouldPaint(const Range& within) const {
  uint32 last_scanline_start =
      (range_.end_time() / kScanLineWidthClocks) * kScanLineWidthClocks;
  Range pf0_right;
  if (tia(TIA::CTRLPF) & 0x01) {
    pf0_right =
        Range(last_scanline_start + 212, last_scanline_start + 212 + 16);
  } else {
    pf0_right =
        Range(last_scanline_start + 148, last_scanline_start + 148 + 16);
  }
  Range pf0_right_intersect(Range::IntersectRanges(within, pf0_right));
  if (!pf0_right_intersect.IsEmpty())
    return pf0_right_intersect.end_time() - 1;

  Range pf0_left(last_scanline_start + 68, last_scanline_start + 68 + 16);
  Range pf0_left_intersect(Range::IntersectRanges(within, pf0_left));
  if (!pf0_left_intersect.IsEmpty())
    return pf0_left_intersect.end_time() - 1;

  return 0;
}

const uint32 State::EarliestPF1CouldPaint(const Range& within) const {
  Range pf1_right_intersect(Range::IntersectRanges(within, Range(84, 116)));
  if (!pf1_right_intersect.IsEmpty())
    return pf1_right_intersect.end_time() - 1;

  Range pf1_left_intersect(Range::IntersectRanges(within, Range(164, 196)));
  if (!pf1_left_intersect.IsEmpty())
    return pf1_left_intersect.end_time() - 1;

  return 0;
}

const uint32 State::EarliestPF2CouldPaint(const Range& within) const {
  Range pf2_right_intersect(Range::IntersectRanges(within, Range(116, 148)));
  if (!pf2_right_intersect.IsEmpty())
    return pf2_right_intersect.end_time() - 1;

  Range pf2_left_intersect(Range::IntersectRanges(within, Range(196, 228)));
  if (!pf2_left_intersect.IsEmpty())
    return pf2_left_intersect.end_time() - 1;

  return 0;
}

const bool State::PlayfieldPaints(uint32 local_clock) const {
  assert(local_clock >= 68);

  if (local_clock < 148 || !(tia(TIA::CTRLPF) & 0x01)) {
    if (local_clock < 84 || (local_clock >= 148 && local_clock < 164)) {
      // PF0 D4 through D7 left to right.
      int pfbit = (local_clock - (local_clock < 84 ? 68 : 148)) >> 2;
      return tia(TIA::PF0) & (0x10 << pfbit);
    } else if (local_clock < 116 || (local_clock >= 164 && local_clock < 196)) {
      // PF1 D7 through D0 left to right.
      int pfbit = (local_clock - (local_clock < 116 ? 84 : 164)) >> 2;
      return tia(TIA::PF1) & (0x80 >> pfbit);
    } else {
      // PF2 D0 through D7 left to right.
      assert(local_clock < 148 || (local_clock >= 196 && local_clock < 228));
      int pfbit = (local_clock - (local_clock < 148 ? 116 : 196)) >> 2;
      return tia(TIA::PF2) & (0x01 << pfbit);
    }
  } else {
    // We are on the right side of the screen and mirroring is turned on.
    assert(local_clock >= 148);
    assert(tia(TIA::CTRLPF) & 0x01);
    if (local_clock < 180) {
      // PF2 D7 through D0 left to right.
      int pfbit = (local_clock - 148) >> 2;
      return tia(TIA::PF2) & (0x80 >> pfbit);
    } else if (local_clock < 212) {
      // PF1 D0 through D7 left to right.
      int pfbit = (local_clock - 180) >> 2;
      return tia(TIA::PF1) & (0x01 << pfbit);
    } else {
      // PF0 D7 through D4 left to right.
      assert(local_clock < 228);
      int pfbit = (local_clock - 212) >> 2;
      return tia(TIA::PF0) & (0x80 >> pfbit);
    }
  }
}

const uint32 State::EarliestBackgroundPaints(const Range& within) const {
  uint32 duration = within.Duration();
  assert(within.end_time() >= duration);
  for (uint32 i = 1; i <= duration; ++i) {
    uint32 color_clock = within.end_time() - i;
    uint32 local_clock = color_clock % kScanLineWidthClocks;
    if (local_clock < kHBlankWidthClocks)
      continue;
    if (PlayerPaints(false, local_clock))
      continue;
    if (PlayerPaints(true, local_clock))
      continue;
    if (PlayfieldPaints(local_clock))
      continue;
    // Neither player nor playfield paints, must be the bg color that paints.
    return color_clock;
  }
  return 0;
}

}  // namespace vcsmc
