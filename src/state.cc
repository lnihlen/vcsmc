#include "state.h"

#include <cassert>
#include <cstring>
#include <iostream>

#include "color.h"
#include "pixel_strip.h"

namespace vcsmc {

State::State()
    : color_clock_(0) {
  std::memset(tia_, 0, sizeof(tia_));
  std::memset(registers_, 0, sizeof(registers_));
}

void State::PaintInto(PixelStrip* pixel_strip, uint32 until) {
  assert(pixel_strip->width() == kFrameWidthPixels * 2);
  assert(color_clock_ < until);
  uint32 local_cc = color_clock_ % kScanLineWidthClocks;
  uint32 local_until = until - color_clock_ + (color_clock_ %
                                               kScanLineWidthClocks);
  uint32 starting_clock = std::max(local_cc, kHBlankWidthClocks);
  uint32 starting_pixel = (starting_clock - kHBlankWidthClocks) * 2;
  for (uint32 clock = starting_clock; clock < local_until; ++clock) {
    uint8 colu = tia_[TIA::COLUBK];
    if (PlayfieldPaints(clock)) {
      colu = tia_[TIA::COLUPF];
    }

    uint32 color = Color::AtariColorToABGR(colu);
    // We paint two pixels for every color clock.
    pixel_strip->SetPixel(starting_pixel++, color);
    pixel_strip->SetPixel(starting_pixel++, color);
  }
}

std::unique_ptr<State> State::Clone() const {
  // Make a copy of ourselves.
  std::unique_ptr<State> state(new State(*this));
  return state;
}

std::unique_ptr<State> State::AdvanceTime(uint32 delta) const {
  std::unique_ptr<State> state(Clone());
  // Add to the color_clock_
  state->color_clock_ += delta;
  return state;
}

std::unique_ptr<State> State::AdvanceTimeAndSetRegister(
    uint32 delta, Register reg, uint8 value) const {
  std::unique_ptr<State> state(AdvanceTime(delta));
  state->registers_[reg] = value;
  return state;
}

std::unique_ptr<State> State::AdvanceTimeAndCopyRegisterToTIA(
    uint32 delta, Register reg, TIA address) const {
  std::unique_ptr<State> state(AdvanceTime(delta));
  uint8 reg_value = state->registers_[reg];
  switch (address) {
    // interesting code here... :)

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
  color_clock_ = state.color_clock_;
}

const bool State::PlayfieldPaints(uint32 clock) {
  assert(clock >= 68);

  // Usage schedule for playfield registers, in color clocks:
  // PF0 68 up until 84
  // PF1 84 - 116
  // PF2 116 - 148
  // PF0 148 - 164
  // PF1 164 - 196
  // PF2 196 - 228

  if (clock < 84 || (clock >= 148 && clock < 164)) {
    // PF0 D4 through D7 left to right.
    int pfbit = (clock - (clock < 84 ? 68 : 148)) >> 2;
    return tia_[TIA::PF0] & (0x10 << pfbit);
  } else if (clock < 116 || (clock >= 164 && clock < 196)) {
    // PF1 D7 through D0 left to right.
    int pfbit = (clock - (clock < 116 ? 84 : 164)) >> 2;
    return tia_[TIA::PF1] & (0x80 >> pfbit);
  } else {
    // PF2 D0 through D7 left to right.
    assert(clock < 148 || (clock >= 196 && clock < 228));
    int pfbit = (clock - (clock < 148 ? 116 : 196)) >> 2;
    return tia_[TIA::PF2] & (0x01 << pfbit);
  }
}

}  // namespace vcsmc
