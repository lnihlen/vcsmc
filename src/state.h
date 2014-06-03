#ifndef SRC_STATE_H_
#define SRC_STATE_H_

#include <memory>
#include <string>

#include "constants.h"
#include "range.h"

namespace vcsmc {

class ColuStrip;
class Spec;

// State represents a moment in time for the VCS. It is a complete (enough for
// our purposes) representation of the VCS hardware. It can generate output
// colu values for all times >= |color_clock_|, which is when this State became
// active.
class State {
 public:
  // Constructs a default initial state, with all values set to unknown.
  State();

  // Fill pixels into |colu_strip| for all time values within |range_|
  void PaintInto(ColuStrip* pixel_strip) const;

  // Returns a color clock value that is the earliest time after which this spec
  // could be added. Returns 0 if it occurs before this State. Note that this
  // method can return times earlier than spec.range().begin_time().
  const uint32 EarliestTimeAfter(const Spec& spec) const;

  //====== State Creation Methods

  // Produces an exact copy of this State.
  std::unique_ptr<State> Clone() const;

  // Given this state at |color_clock_| = t, produce a new State which is an
  // exact copy of this one but at color_clock_ = t + delta time. ALSO HAS THE
  // SIDE EFFECT OF MODIFYING OUR OWN TIME RANGE.
  std::unique_ptr<State> AdvanceTime(uint32 delta);

  // Same as above, but the returned state also has new value stored in reg.
  std::unique_ptr<State> AdvanceTimeAndSetRegister(uint32 delta,
                                                   Register reg,
                                                   uint8 value);

  // Same as above, but the returned state also simulates the effect of copying
  // the value in reg to the supplied tia address. Note that for some of the
  // strobe values this can have substantial impact on other states within the
  // new state. Also note that setting some states is invalid, like strobing
  // HMOVE at times other than at the start of HBLANK, or in general setting
  // values while they are in use, and if such a state is invalid this function
  // will return nullptr.
  std::unique_ptr<State> AdvanceTimeAndCopyRegisterToTIA(uint32 delta,
                                                         Register reg,
                                                         TIA address);
  //====== Utility Methods

  // Given a value like Register::A returns "a";
  static std::string RegisterToString(const Register reg);
  // Given a ZeroPage address returns either a human-readable name, if within
  // the TIA realm, or a hexadecimal number for the address.
  static std::string AddressToString(const uint8 address);
  // Really only here because the other things are here. Given 0xfe will return
  // the string "$fe".
  static std::string ByteToHexString(const uint8 value);

  const uint8 a() const { return registers_[Register::A]; }
  const uint8 x() const { return registers_[Register::X]; }
  const uint8 y() const { return registers_[Register::Y]; }
  const uint8 tia(TIA address) const { return tia_[address]; }
  const Range& range() const { return range_; }
  const bool register_known(Register reg) const {
    return registers_known_ & (1 << static_cast<int>(reg));
  }
  const bool tia_known(TIA address) const {
    return tia_known_ & (1 << static_cast<int>(address));
  }

 private:
  State(const State& state);

  // lcoal_clock is [0.. 228)
  // Returns the earliest time the player actually renders a pixel.
  const uint32 EarliestPlayerPaints(bool p1, const Range& within) const;
  // Returns the earliest time the player _could_ render a pixel, meaning the
  // earliest time that the player bitfield is actually be considered for
  // render.
  const uint32 EarliestPlayerCouldPaint(bool p1, const Range& within) const;
  const bool PlayerPaints(bool p1, uint32 local_clock) const;
  const bool PlayerCouldPaint(bool p1, uint32 local_clock) const;

  const uint32 EarliestPlayfieldPaints(const Range& within) const;
  const uint32 EarliestPF0CouldPaint(const Range& within) const;
  const uint32 EarliestPF1CouldPaint(const Range& within) const;
  const uint32 EarliestPF2CouldPaint(const Range& within) const;
  const bool PlayfieldPaints(uint32 local_clock) const;

  const uint32 EarliestBackgroundPaints(const Range& within) const;

  uint8 tia_[TIA::TIA_COUNT];
  uint64 tia_known_;
  uint8 registers_[Register::REGISTER_COUNT];
  uint8 registers_known_;
  Range range_;
};

}  // namespace vcsmc

#endif  // SRC_STATE_H_
