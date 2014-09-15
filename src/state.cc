#include "state.h"

#include <cstring>
#include <iostream>

#include "color.h"
#include "colu_strip.h"
#include "spec.h"

namespace {

// playfield schedules
// +-----------+------+---------------------+--------------------+
// | playfield | left | right mirroring off | right mirroring on |
// +-----------+------+---------------------+--------------------+
// |    PF0    |  68  |        148          |        212         |
// |    PF1    |  84  |        164          |        180         |
// |    PF2    | 116  |        196          |        148         |
// +-----------+------+---------------------+--------------------+
const uint64 kPF0Left = 68;
const uint64 kPF1Left = 84;
const uint64 kPF2Left = 116;
const uint64 kPF0Right = 148;
const uint64 kPF1Right = 164;
const uint64 kPF2Right = 196;
const uint64 kPF0RightM = 212;
const uint64 kPF1RightM = 180;
const uint64 kPF2RightM = 148;
const uint64 kPFMidline = 148;

}


namespace vcsmc {

State::State()
    : tia_known_(kTIAStrobeMask),
      registers_known_(0),
      range_(0, kFrameSizeClocks) {
  // The unknown/known logic asserts on |tia_| or |registers_| access, so we
  // can skip initialization of their memory areas.
}

State::State(const uint8* tia_values)
    : tia_known_(0xffffffffffffffff),
      registers_known_(0),
      range_(0, kFrameSizeClocks) {
  std::memcpy(tia_, tia_values, sizeof(tia_));
}

std::unique_ptr<State> State::Clone() const {
  std::unique_ptr<State> state(new State(*this));
  return state;
}

std::unique_ptr<State> State::AdvanceTime(uint64 delta) {
  assert(delta > 0);
  std::unique_ptr<State> state(Clone());
  uint64 new_start_time = range_.start_time() + delta;
  range_.set_end_time(new_start_time);
  // The new |state| has a copy of our |range_|, which may have had an end
  // time less than the new start time. If so, we reset the new |state| to
  // have an empty range starting at |new_start_time|.
  if (state->range_.end_time() < new_start_time)
    state->range_.set_end_time(new_start_time);
  state->range_.set_start_time(new_start_time);
  return state;
}

std::unique_ptr<State> State::AdvanceTimeAndSetRegister(
    uint64 delta, Register axy, uint8 value) {
  std::unique_ptr<State> state(AdvanceTime(delta));
  state->registers_known_ |= (1 << static_cast<int>(axy));
  state->registers_[axy] = value;
  return state;
}

std::unique_ptr<State> State::AdvanceTimeAndCopyRegisterToTIA(
    uint64 delta, Register axy, TIA address) {
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

std::unique_ptr<State> State::MakeEntryState(uint64 delta) {
  std::unique_ptr<State> state;
  if (delta > 0) {
    state = AdvanceTime(delta);
    state->range_ = Range(range_.end_time(), range_.end_time());
  } else {
    state = Clone();
    state->range_ = Range(range_.start_time(), range_.start_time());
  }
  state->registers_known_ = 0;
  return state;
}

void State::PaintInto(ColuStrip* colu_strip) const {
  Range strip_range(Range::IntersectRanges(colu_strip->range(), range_));
  uint64 local_clock = strip_range.start_time() % kScanLineWidthClocks;
  uint64 local_until = local_clock + strip_range.Duration();
  uint64 starting_clock = std::max(local_clock, kHBlankWidthClocks);
  uint64 starting_column = starting_clock - kHBlankWidthClocks;
  for (uint64 clock = starting_clock; clock < local_until; ++clock) {
    uint8 colu = tia(TIA::COLUBK);
    if (PlayfieldPaints(clock)) {
      // If D1 of CTRLPF is set the playfield paints with COLUP0 on the left
      // side and COLUP1 on the right side.
      if (tia(TIA::CTRLPF) & 0x02)
        colu = clock < kPFMidline ? tia(TIA::COLUP0) : tia(TIA::COLUP1);
      else
        colu = tia(TIA::COLUPF);
    }

    colu_strip->set_colu(starting_column++, colu);
  }
}

uint64 State::EarliestTimeAfter(const Spec& spec) const {
  return EarliestTimeAfterWithEndTime(spec, range_.end_time());
}

uint64 State::EarliestTimeAfterWithEndTime(
    const Spec& spec, uint64 end_time) const {
  assert(end_time >= range_.start_time());
  Range range(range_.start_time(), end_time);
  Range within(Range::IntersectRanges(range, spec.range()));
  if (within.IsEmpty())
    return kInfinity;

  uint64 state_earliest = kInfinity;

  switch(spec.tia()) {
    // Not supported currently.
    case TIA::VSYNC:
    case TIA::VBLANK:
    case TIA::WSYNC:
    case TIA::RSYNC:
      break;

    // NUSIZ support coming later.
    case TIA::NUSIZ0:
    case TIA::NUSIZ1:
      break;

    case TIA::COLUP0:
    case TIA::COLUP1:
      break;

    case TIA::COLUPF:
      // If in score mode we ignore COLUPF for all playfield rendering.
      if (tia(TIA::CTRLPF) & 0x02)
         state_earliest = 0;
      else
        state_earliest = EarliestPlayfieldPaints(range);
      break;

    case TIA::COLUBK:
      state_earliest = EarliestBackgroundPaints(range);
      break;

    case TIA::CTRLPF:
      state_earliest = EarliestTimeInHBlank(range);
      break;

    case TIA::REFP0:
    case TIA::REFP1:
      break;

    case TIA::PF0:
    case TIA::PF1:
    case TIA::PF2:
      state_earliest = EarliestPFXCouldPaint(spec.tia(), range);
      break;

    // These are strobe registers and need to happen right when they need to
    // happen, so are not subject to back-scheduling. Some thought to be given
    // about how to handle timing on these.
    case TIA::RESP0:
    case TIA::RESP1:
    case TIA::RESM0:
    case TIA::RESM1:
    case TIA::RESBL:
      break;

    // Audio registers updated at fixed intervals and not supported by
    // graphics scheduler.
    case TIA::AUDC0:
    case TIA::AUDC1:
    case TIA::AUDF0:
    case TIA::AUDF1:
    case TIA::AUDV0:
    case TIA::AUDV1:
      break;

    case TIA::GRP0:
    case TIA::GRP1:
      break;

    // Ball and missile support coming later.
    case TIA::ENAM0:
    case TIA::ENAM1:
    case TIA::ENABL:
      break;

    // Some tricky timing considerations here. Leave alone for now but when
    // adding player supprt this will need revision. These can be written any
    // time after a prior strobe of HMOVE.
    case TIA::HMP0:
    case TIA::HMP1:
      break;

    // Pending missile and ball support.
    case TIA::HMM0:
    case TIA::HMM1:
    case TIA::HMBL:
      break;

    // While potentially quite powerful we will ignore these for now.
    case TIA::VDELP0:
    case TIA::VDELP1:
      break;

    // Pending missile and ball support.
    case TIA::VDELBL:
    case TIA::RESMP0:
    case TIA::RESMP1:
      break;

    // Another strobe with zero scheduling flexibility.
    case TIA::HMOVE:
      break;

    // Potential conflicts with setting any of the other horizontal motion
    // registers, but otherwise represents an efficient way to get back to a
    // known state.
    case TIA::HMCLR:
      break;

    // Non-interactive software has no use for a collision latch that I can
    // think of.
    case TIA::CXCLR:
      break;

    case TIA::TIA_COUNT:
      break;
  }

  if (state_earliest == kInfinity ||
      (state_earliest >= within.end_time() - 1))
    return kInfinity;

  if (state_earliest == 0 && (spec.range().start_time() < range.start_time()))
    return 0;

  return std::max(state_earliest, within.start_time());
}

State::State(const State& state) {
  std::memcpy(tia_, state.tia_, TIA_COUNT);
  std::memcpy(registers_, state.registers_, REGISTER_COUNT);
  tia_known_ = state.tia_known_;
  registers_known_ = state.registers_known_;
  range_ = state.range_;
}

const bool State::PlayfieldPaints(uint64 local_clock) const {
  assert(local_clock >= kPF0Left);

  if (local_clock < kPFMidline || !(tia(TIA::CTRLPF) & 0x01)) {
    if (local_clock < kPF1Left ||
        (local_clock >= kPF0Right && local_clock < kPF1Right)) {
      // PF0 D4 through D7 left to right.
      int pfbit =
          (local_clock - (local_clock < kPF1Left ? kPF0Left : kPF0Right)) >> 2;
      return tia(TIA::PF0) & (0x10 << pfbit);
    } else if (local_clock < kPF2Left ||
        (local_clock >= kPF1Right && local_clock < kPF2Right)) {
      // PF1 D7 through D0 left to right.
      int pfbit =
          (local_clock - (local_clock < kPF2Left ? kPF1Left : kPF1Right)) >> 2;
      return tia(TIA::PF1) & (0x80 >> pfbit);
    } else {
      // PF2 D0 through D7 left to right.
      assert(local_clock < kPF0Right ||
          (local_clock >= kPF2Right && local_clock < 228));
      int pfbit =
          (local_clock - (local_clock < kPF0Right ? kPF2Left : kPF2Right)) >> 2;
      return tia(TIA::PF2) & (0x01 << pfbit);
    }
  } else {
    // We are on the right side of the screen and mirroring is turned on.
    assert(local_clock >= kPFMidline);
    assert(tia(TIA::CTRLPF) & 0x01);
    if (local_clock < kPF1RightM) {
      // PF2 D7 through D0 left to right.
      int pfbit = (local_clock - kPF2RightM) >> 2;
      return tia(TIA::PF2) & (0x80 >> pfbit);
    } else if (local_clock < kPF0RightM) {
      // PF1 D0 through D7 left to right.
      int pfbit = (local_clock - kPF1RightM) >> 2;
      return tia(TIA::PF1) & (0x01 << pfbit);
    } else {
      // PF0 D7 through D4 left to right.
      assert(local_clock < 228);
      int pfbit = (local_clock - kPF0RightM) >> 2;
      return tia(TIA::PF0) & (0x80 >> pfbit);
    }
  }
}

const uint64 State::EarliestPlayfieldPaints(const Range& range) const {
  uint64 duration = range.Duration();
  for (uint32 i = 1; i <= duration; ++i) {
    uint64 color_clock = range.end_time() - i;
    uint64 local_clock = color_clock % kScanLineWidthClocks;
    if (local_clock < kHBlankWidthClocks)
      continue;

    if (PlayfieldPaints(local_clock))
      return color_clock;
  }
  return 0;
}

const uint64 State::EarliestPFXCouldPaint(TIA pf, const Range& range) const {
  uint64 last_scanline_start =
      (range.end_time() / kScanLineWidthClocks) * kScanLineWidthClocks;
  uint64 right_pf_begin = 0;
  uint64 right_pf_end = 0;
  if (tia(TIA::CTRLPF) & 0x01) {
    switch (pf) {
      case TIA::PF0:
        right_pf_begin = kPF0RightM;
        right_pf_end = kPF0RightM + 16;
        break;

      case TIA::PF1:
        right_pf_begin = kPF1RightM;
        right_pf_end = kPF1RightM + 32;
        break;

      case TIA::PF2:
        right_pf_begin = kPF2RightM;
        right_pf_end = kPF2RightM + 32;
        break;

      default:
        assert(false);
        break;
    }
  } else {
    switch (pf) {
      case TIA::PF0:
        right_pf_begin = kPF0Right;
        right_pf_end = kPF0Right + 16;
        break;

      case TIA::PF1:
        right_pf_begin = kPF1Right;
        right_pf_end = kPF1Right + 32;
        break;

      case TIA::PF2:
        right_pf_begin = kPF2Right;
        right_pf_end = kPF2Right + 32;
        break;

      default:
        assert(false);
        break;
    }
  }
  Range pfx_right(last_scanline_start + right_pf_begin,
      last_scanline_start + right_pf_end);
  Range pfx_right_intersect(Range::IntersectRanges(range, pfx_right));
  if (!pfx_right_intersect.IsEmpty())
    return pfx_right_intersect.end_time() - 1;

  uint64 left_pf_begin = 0;
  uint64 left_pf_end = 0;
  switch (pf) {
    case TIA::PF0:
      left_pf_begin = kPF0Left;
      left_pf_end = kPF0Left + 16;
      break;

    case TIA::PF1:
      left_pf_begin = kPF1Left;
      left_pf_end = kPF1Left + 32;
      break;

    case TIA::PF2:
      left_pf_begin = kPF2Left;
      left_pf_end = kPF2Left + 32;
      break;

    default:
      assert(false);
      break;
  }
  Range pfx_left(last_scanline_start + left_pf_begin,
      last_scanline_start + left_pf_end);
  Range pfx_left_intersect(Range::IntersectRanges(range, pfx_left));
  if (!pfx_left_intersect.IsEmpty())
    return pfx_left_intersect.end_time() - 1;

  // It's possible that the range spans to the previous scanline, and could
  // intersect with the right-side PFX there.
  if (range.start_time() < last_scanline_start && last_scanline_start > 0) {
    uint64 prev_scanline_start = last_scanline_start - kScanLineWidthClocks;
    Range pfx_prev_right(prev_scanline_start + right_pf_begin,
        prev_scanline_start + right_pf_end);
    Range pfx_prev_right_intersect(
        Range::IntersectRanges(range, pfx_prev_right));
    if (!pfx_prev_right_intersect.IsEmpty())
      return pfx_prev_right_intersect.end_time() - 1;
  }

  return 0;
}

const uint64 State::EarliestBackgroundPaints(const Range& range) const {
  uint64 duration = range.Duration();
  for (uint64 i = 1; i <= duration; ++i) {
    uint64 color_clock = range.end_time() - i;
    uint64 local_clock = color_clock % kScanLineWidthClocks;
    if (local_clock < kHBlankWidthClocks)
      continue;
    if (PlayfieldPaints(local_clock))
      continue;
    // Neither player nor playfield paints, must be the bg color that paints.
    return color_clock;
  }
  return 0;
}

const uint64 State::EarliestTimeInHBlank(const Range& range) const {
  uint64 last_scanline_start =
      (range.end_time() / kScanLineWidthClocks) * kScanLineWidthClocks;
  Range last_scanline_hblank(last_scanline_start,
      last_scanline_start + kHBlankWidthClocks);
  Range section(Range::IntersectRanges(range, last_scanline_hblank));

  // If we have no time within the HBlank of our last line return kInfinity.
  if (section.IsEmpty())
    return kInfinity;

  // If we have time after the HBlank ends we cannot schedule before.
  if (last_scanline_hblank.end_time() < range.end_time())
    return kInfinity;

  // If HBlank begins before we begin we can return 0, otherwise we return the
  // start of the HBlank.
  return last_scanline_start < range.start_time() ? 0 : last_scanline_start;
}

}  // namespace vcsmc
