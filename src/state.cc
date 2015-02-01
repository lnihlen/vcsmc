#include "state.h"

#include <cstring>
#include <iostream>

#include "cl_image.h"
#include "color.h"
#include "image.h"
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
const uint32 kPF0Left = 68;
const uint32 kPF1Left = 84;
const uint32 kPF2Left = 116;
const uint32 kPF0Right = 148;
const uint32 kPF1Right = 164;
const uint32 kPF2Right = 196;
const uint32 kPF0RightM = 212;
const uint32 kPF1RightM = 180;
const uint32 kPF2RightM = 148;
const uint32 kPFMidline = 148;

}

namespace vcsmc {

State::State()
    : tia_known_(kTIAStrobeMask),
      registers_known_(0),
      range_(0, kScreenSizeClocks),
      p0_clock_(0),
      p0_bit_(kInfinity),
      p1_clock_(0),
      p1_bit_(kInfinity) {
  // The unknown/known logic asserts on |tia_| or |registers_| access, so we
  // can skip initialization of their memory areas.
}

State::~State() {
}

State::State(const uint8* tia_values)
    : tia_known_(0xffffffffffffffff),
      registers_known_(0),
      range_(0, kScreenSizeClocks),
      p0_clock_(0),
      p0_bit_(kInfinity),
      p1_clock_(0),
      p1_bit_(kInfinity) {
  std::memcpy(tia_, tia_values, sizeof(tia_));
}

std::unique_ptr<State> State::Clone() const {
  std::unique_ptr<State> state(new State(*this));
  return state;
}

std::unique_ptr<State> State::AdvanceTime(uint32 delta) {
  std::unique_ptr<State> state(Clone());
  uint32 new_start_time = range_.start_time() + delta;
  range_.set_end_time(new_start_time);
  // The new |state| has a copy of our |range_|, which may have had an end
  // time less than the new start time. If so, we reset the new |state| to
  // have an empty range starting at |new_start_time|.
  if (state->range_.end_time() < new_start_time)
    state->range_.set_end_time(new_start_time);
  state->range_.set_start_time(new_start_time);

  state->p0_clock_ += delta;
  if (state->p0_bit_ != kInfinity) {
    state->p0_bit_ += delta;
    if (state->p0_bit_ >= 13)
      state->p0_bit_ = kInfinity;
  }

  state->p1_clock_ += delta;
  if (state->p1_bit_ != kInfinity) {
    state->p1_bit_ += delta;
    if (state->p1_bit_ >= 13)
      state->p1_bit_ = kInfinity;
  }

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
  if (address == TIA::WSYNC) {
    delta = (((range_.start_time() / kScanLineWidthClocks) + 1) *
        kScanLineWidthClocks) - range_.start_time();
  }
  std::unique_ptr<State> state(AdvanceTime(delta));
  uint8 reg_value = state->reg(axy);
  state->SetTIA(address, reg_value);
  return state;
}

std::unique_ptr<State> State::MakeEntryState(uint32 delta) {
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

std::unique_ptr<State> State::MakeIdealState(const Spec& spec) {
  assert(spec.range().start_time() >= range_.start_time());
  uint32 delta = spec.range().start_time() - range_.start_time();
  std::unique_ptr<State> state(AdvanceTime(delta));
  state->SetTIA(spec.tia(), spec.value());
  return state;
}

void State::PaintInto(Image* image) const {
  // Cull any states outside of the range of painting scan lines.
  Range paint_range = Range::IntersectRanges(Range(
      (kVSyncScanLines + kVBlankScanLines) * kScanLineWidthClocks,
      (kVSyncScanLines + kVBlankScanLines + kFrameHeightPixels)
          * kScanLineWidthClocks), range_);
  if (paint_range.IsEmpty())
    return;

  for (uint32 i = paint_range.start_time(); i < paint_range.end_time(); ++i) {
    uint32 clock = i % kScanLineWidthClocks;
    if (clock < kHBlankWidthClocks)
      continue;
    uint8 colu = ColuForClock(i);
    uint32 x = clock - kHBlankWidthClocks;
    uint32 y = (i / kScanLineWidthClocks) - kVSyncScanLines - kVBlankScanLines;
    *(image->pixels_writeable() + (y * kFrameWidthPixels) + x) =
        Color::AtariColorToABGR(colu);
  }
}

void State::ColorInto(uint8* colus, const Range& range) const {
  // Cull any states outside of the range provided.
  Range paint_range = Range::IntersectRanges(range, range_);
  paint_range = Range::IntersectRanges(paint_range, Range(
      (kVSyncScanLines + kVBlankScanLines) * kScanLineWidthClocks,
      (kVSyncScanLines + kVBlankScanLines + kFrameHeightPixels)
          * kScanLineWidthClocks));
  if (paint_range.IsEmpty())
    return;

  for (uint32 i = paint_range.start_time(); i < paint_range.end_time(); ++i) {
    uint32 clock = i % kScanLineWidthClocks;
    if (clock < kHBlankWidthClocks)
      continue;
    uint8 colu = ColuForClock(i) / 2;
    uint32 x = clock - kHBlankWidthClocks;
    uint32 y = (i / kScanLineWidthClocks) - kVSyncScanLines - kVBlankScanLines;
    *(colus + (y * kFrameWidthPixels) + x) = colu;
  }
}

State::State(const State& state) {
  std::memcpy(tia_, state.tia_, TIA_COUNT);
  std::memcpy(registers_, state.registers_, REGISTER_COUNT);
  tia_known_ = state.tia_known_;
  registers_known_ = state.registers_known_;
  range_ = state.range_;
  p0_clock_ = state.p0_clock_;
  p0_bit_ = state.p0_bit_;
  p1_clock_ = state.p1_clock_;
  p1_bit_ = state.p1_bit_;
}

void State::SetTIA(TIA address, uint8 value) {
  uint32 player_clock = 0;
  switch (address) {
    case TIA::RESP0:
      player_clock = p0_clock_ % kScanLineWidthClocks;
      if (p0_clock_ > kScanLineWidthClocks &&
          player_clock < 13) {
        p0_bit_ = player_clock;
      } else {
        p0_bit_ = kInfinity;
      }
      p0_clock_ = 0;
      break;

    case TIA::RESP1:
      player_clock = p1_clock_ % kScanLineWidthClocks;
      if (p1_clock_ > kScanLineWidthClocks &&
          player_clock < 13) {
        p1_bit_ = player_clock;
      } else {
        p1_bit_ = kInfinity;
      }
      p1_clock_ = 0;
      break;

    default:
      tia_known_ |= (1 << static_cast<int>(address));
      tia_[address] = value;
      break;
  }
}

uint8 State::ColuForClock(uint32 clock) const {
  uint8 colu = tia(TIA::COLUBK);
  uint32 local_clock = clock % kScanLineWidthClocks;
  if (PlayerPaints(clock, false)) {
    colu = tia(TIA::COLUP0);
  } else if (PlayerPaints(clock, true)) {
    colu = tia(TIA::COLUP1);
  } else if (PlayfieldPaints(local_clock)) {
    // If D1 of CTRLPF is set the playfield paints with COLUP0 on the left
    // side and COLUP1 on the right side.
    if (tia(TIA::CTRLPF) & 0x02) {
      colu = clock < kPFMidline ? tia(TIA::COLUP0) : tia(TIA::COLUP1);
    } else {
      colu = tia(TIA::COLUPF);
    }
  }
  return colu;
}

bool State::PlayfieldPaints(uint32 local_clock) const {
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

bool State::PlayerPaints(uint32 global_clock, bool is_player_one) const {
  // Early-out for player graphics disabled.
  TIA grpx = is_player_one ? TIA::GRP1 : TIA::GRP0;
  if (!tia(grpx))
    return false;

  assert(global_clock >= range_.start_time());
  uint32 offset_from_start = global_clock - range_.start_time();
  uint32 px_bit = is_player_one ? p1_bit_ : p0_bit_;
  if (px_bit != kInfinity &&
      px_bit + offset_from_start >= 5 &&
      px_bit + offset_from_start < 13)
    return (tia(grpx) & (1 << (px_bit + offset_from_start - 5)));

  uint32 player_clock = is_player_one ? p1_clock_ : p0_clock_;
  player_clock += offset_from_start;
  if (player_clock < kScanLineWidthClocks)
    return false;

  uint32 pixel_clock = player_clock % kScanLineWidthClocks;
  if (pixel_clock >= 5 && pixel_clock < 13) {
    uint32 bit = pixel_clock - 5;
    return (tia(grpx) & (1 << bit));
  }

  return false;
}

}  // namespace vcsmc
