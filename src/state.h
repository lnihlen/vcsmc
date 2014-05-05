#ifndef SRC_STATE_H_
#define SRC_STATE_H_

#include <memory>
#include <string>

#include "colu_strip.h"
#include "constants.h"

namespace vcsmc {

// State represents a moment in time for the VCS. It is a complete (enough for
// our purposes) representation of the VCS hardware. It can generate output
// colu values for all times >= |color_clock_|, which is when this State became
// active.
class State {
 public:
  // Constructs a default initial state, with all values set to zero.
  State();
  ~State();

  // Defines the address and name of every register on the TIA. The ones marked
  // as (strobe) are write-only and writing to them will cause new changes in
  // state.
  enum TIA : uint8 {
    VSYNC  = 0x00,  // vertical sync set-clear
    VBLANK = 0x01,  // vertical blank set-clear
    WSYNC  = 0x02,  // (strobe) wait for leading edge of horizontal blank
    RSYNC  = 0x03,  // (strobe) reset horizontal sync counter
    NUSIZ0 = 0x04,  // number-size player-missile 0
    NUSIZ1 = 0x05,  // number-size player-missile 1
    COLUP0 = 0x06,  // color-lum player 0
    COLUP1 = 0x07,  // colur-lum player 1
    COLUPF = 0x08,  // colur-lum playfield
    COLUBK = 0x09,  // colur-lum background
    CTRLPF = 0x0a,  // control playfield ball size & collisions
    REFP0  = 0x0b,  // reflect player 0
    REFP1  = 0x0c,  // reflect player 1
    PF0    = 0x0d,  // playfield register byte 0
    PF1    = 0x0e,  // playfield register byte 1
    PF2    = 0x0f,  // playfield register byte 2
    RESP0  = 0x10,  // (strobe) reset player 0
    RESP1  = 0x11,  // (strobe) reset player 1
    RESM0  = 0x12,  // (strobe) reset missile 0
    RESM1  = 0x13,  // (strobe) reset missile 1
    RESBL  = 0x14,  // (strobe) reset ball
    AUDC0  = 0x15,  // audio control 0
    AUDC1  = 0x16,  // audio control 1
    AUDF0  = 0x17,  // audio frequency 0
    AUDF1  = 0x18,  // audio frequency 1
    AUDV0  = 0x19,  // audio volume 0
    AUDV1  = 0x1a,  // audio volume 1
    GRP0   = 0x1b,  // graphics player 0
    GRP1   = 0x1c,  // graphics player 1
    ENAM0  = 0x1d,  // graphics (enable) missile 0
    ENAM1  = 0x1e,  // graphics (enable) missile 1
    ENABL  = 0x1f,  // graphics (enable) ball
    HMP0   = 0x20,  // horizontal motion player 0
    HMP1   = 0x21,  // horizontal motion player 1
    HMM0   = 0x22,  // horizontal motion missile 0
    HMM1   = 0x23,  // horizontal motion missile 1
    HMBL   = 0x24,  // horizontal motion ball
    VDELP0 = 0x25,  // vertical delay player 0
    VDELP1 = 0x26,  // vertical delay player 1
    VDELBL = 0x27,  // vertical delay ball
    RESMP0 = 0x28,  // reset missile 0 to player 0
    RESMP1 = 0x29,  // reset missile 1 to player 1
    HMOVE  = 0x2a,  // (strobe) apply horizontal motion
    HMCLR  = 0x2b,  // (strobe) clear horizontal motion registers
    CXCLR  = 0x2c,  // (strobe) clear collision latches
    TIA_COUNT  = 0x2d
  };

  enum Register : uint8 {
    A = 0,
    Y = 1,
    X = 2,
    REGISTER_COUNT = 3
  };

  // Fill pixels into |colu_strip| from [color_clock, until)
  void PaintInto(ColuStrip* colu_strip, uint32 until);

  //====== State Creation Methods

  // Produces an exact copy of this State.
  std::unique_ptr<State> Clone() const;

  // Given this state at |color_clock_| = t, produce a new State which is an
  // exact copy of this one but at color_clock_ = t + delta time.
  std::unique_ptr<State> AdvanceTime(uint32 delta) const;

  // Same as above, but the returned state also has new value stored in reg.
  std::unique_ptr<State> AdvanceTimeAndSetRegister(uint32 delta,
                                                   Register reg,
                                                   uint8 value) const;

  // Save as above, but the returned state also simulates the effect of copying
  // the value in reg to the supplied tia address. Note that for some of the
  // strobe values this can have substantial impact on other states within the
  // new state. Also note that setting some states is invalid, like strobing
  // HMOVE at times other than at the start of HBLANK, or in general setting
  // values while they are in use, and if such a state is invalid this function
  // will return nullptr.
  std::unique_ptr<State> AdvanceTimeAndCopyRegisterToTIA(uint32 delta,
                                                         Register reg,
                                                         TIA address) const;
  //====== Utility Methods

  // Given a value like Register::A returns "a";
  static std::string RegisterToString(const Register reg);
  // Given a ZeroPage address returns either a human-readable name, if within
  // the TIA realm, or a hexadecimal number for the address.
  static std::string AddressToString(const uint8 address);
  // Really only here because the other things are here. Given 0xfe will return
  // the string "$fe".
  static std::string ByteToHexString(const uint8 value);

  const uint32 color_clocks() const { return color_clock_; }
  const uint8 x() const { return registers_[Register::X]; }
  const uint8 y() const { return registers_[Register::Y]; }
  const uint8 a() const { return registers_[Register::A]; }
  const uint8 tia(TIA address) const { return tia_[address]; }

 private:
  State(const State& state);
  uint8 tia_[TIA_COUNT];
  uint8 registers_[REGISTER_COUNT];
  uint32 color_clock_;
};

}  // namespace vcsmc

#endif  // SRC_STATE_H_
