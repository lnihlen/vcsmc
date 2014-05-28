#ifndef SRC_STATE_H_
#define SRC_STATE_H_

#include <memory>
#include <string>

#include "constants.h"

namespace vcsmc {

class ColuStrip;

// State represents a moment in time for the VCS. It is a complete (enough for
// our purposes) representation of the VCS hardware. It can generate output
// colu values for all times >= |color_clock_|, which is when this State became
// active.
class State {
 public:
  // Constructs a default initial state, with all values set to zero.
  State();

  // Fill pixels into |colu_strip| from [color_clock, until)
  void PaintInto(ColuStrip* pixel_strip, uint32 until);

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
  const uint8 a() const { return registers_[Register::A]; }
  const uint8 x() const { return registers_[Register::X]; }
  const uint8 y() const { return registers_[Register::Y]; }
  const uint8 tia(TIA address) const { return tia_[address]; }

 private:
  State(const State& state);
  // Clock in this case is the lcoal_clock, that is [0..228)
  const bool PlayfieldPaints(uint32 clock);
  uint8 tia_[TIA::TIA_COUNT];
  uint8 registers_[Register::REGISTER_COUNT];
  uint32 color_clock_;
};

}  // namespace vcsmc

#endif  // SRC_STATE_H_
