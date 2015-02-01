#ifndef SRC_STATE_H_
#define SRC_STATE_H_

#include <cassert>
#include <memory>
#include <string>

#include "constants.h"
#include "range.h"

namespace vcsmc {

class Image;
class Spec;

// State represents a moment in time for the VCS. It is a complete (enough for
// our purposes) representation of the VCS hardware. It can generate output
// colu values for all times >= |color_clock_|, which is when this State became
// active.
class State {
 public:
  //====== State Creation Methods

  // Constructs a default initial State, with all values set to unknown.
  State();
  ~State();

  // Constructs a State with all TIA values known and set to a copy of the
  // provided byte pointer, all registers set to unknown, and all over values
  // at defaults.
  State(const uint8* tia_values);

  // Produces an exact copy of this State.
  std::unique_ptr<State> Clone() const;

  // Given this state at |color_clock_| = t, produce a new State which is an
  // exact copy of this one but at color_clock_ = t + delta time. ALSO HAS THE
  // SIDE EFFECT OF MODIFYING OUR OWN TIME RANGE.
  std::unique_ptr<State> AdvanceTime(uint32 delta);

  // Same as above, but the returned state also has new value stored in reg.
  std::unique_ptr<State> AdvanceTimeAndSetRegister(uint32 delta,
                                                   Register axy,
                                                   uint8 value);

  // Same as above, but the returned state also simulates the effect of copying
  // the value in reg to the supplied tia address. Note that for some of the
  // strobe values this can have substantial impact on other states within the
  // new state.
  std::unique_ptr<State> AdvanceTimeAndCopyRegisterToTIA(uint32 delta,
                                                         Register axy,
                                                         TIA address);

  // Returns a new State that is a copy of this one but with empty |range_|
  // and with all register values reset to unknown. Modifies this State's
  // range().end_time() just like AdvanceTime(delta) but in this case |delta|
  // may also be zero. If Delta is zero the entry state will have an empty
  // range starting at our start_time(), otherwise it will have an empty range
  // starting at this State's end_time().
  std::unique_ptr<State> MakeEntryState(uint32 delta);

  // Adapt whatever change ordered by |spec| to a new state starting at the
  // start time of the range of the |spec| and return.
  std::unique_ptr<State> MakeIdealState(const Spec& spec);

  //====== Simulation and Scheduling

   // Fill pixels in |image| for all time values within |range_|.
  void PaintInto(Image* image) const;

  // Fill bytes in |colus| with _half_ atari color values within |range|.
  void ColorInto(uint8* colus, const Range& range) const;

  const bool register_known(Register axy) const {
    return registers_known_ & (1 << static_cast<int>(axy));
  }
  const bool tia_known(TIA address) const {
    return tia_known_ & (1ULL << static_cast<int>(address));
  }
  const uint8 a() const {
    assert(register_known(Register::A));
    return registers_[Register::A];
  }
  const uint8 x() const {
    assert(register_known(Register::X));
    return registers_[Register::X];
  }
  const uint8 y() const {
    assert(register_known(Register::Y));
    return registers_[Register::Y];
  }
  const uint8 reg(Register axy) const {
    assert(register_known(axy));
    return registers_[axy];
  }
  const uint8 tia(TIA address) const {
    assert(tia_known(address));
    return tia_[address];
  }
  const Range& range() const { return range_; }

 private:
  State(const State& s);
  void SetTIA(TIA address, uint8 value);

  uint8 ColuForClock(uint32 clock) const;
  bool PlayfieldPaints(uint32 local_clock) const;
  bool PlayerPaints(uint32 global_clock, bool is_player_one) const;

  uint8 tia_[TIA::TIA_COUNT];
  uint64 tia_known_;
  uint8 registers_[Register::REGISTER_COUNT];
  uint8 registers_known_;
  Range range_;
  uint32 p0_clock_;
  uint32 p0_bit_;
  uint32 p1_clock_;
  uint32 p1_bit_;
};

}  // namespace vcsmc

#endif  // SRC_STATE_H_
