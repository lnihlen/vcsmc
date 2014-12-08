#ifndef SRC_SCHEDULE_H_
#define SRC_SCHEDULE_H_

#include <memory>
#include <string>
#include <vector>

#include "constants.h"
#include "types.h"

namespace vcsmc {

class Block;
class Spec;
class State;
class TimeSpan;

// A Schedule is our model of 6502 bytecode and time to produce the target
// image. It consists of Blocks of OpCodes as well as their output states. It
// also maintains an array of TIA state which can be appended to a Block for the
// output of an initialization block, which is assumed to run with during the
// vertical blanking time of the frame, so no concerns about scheduling need
// apply. Schedule's job is to provide data on cost to add Specs issued by
// Strategies, and to support actually adding those Specs that can be scheduled.
// Once all scan lines have been fit Schedule can output the final program in
// either 6502 bytecode or assembly text, for ease of debugging.

// Schedules support lazy copying and tentative adds. Kernel keeps a master
// Schedule. When trying a Strategy for a new scan line it creates lazy copies
// of the master schedule. A lazy copy is made by using the LazyBlock object,
// which keeps a shared_ptr<Block>. All non-modifying block operations call
// through to the shared_ptr. When a modifying block operation occurs, such as
// Block::Append(), a copy of the shared_ptr is made and the change is applied
// to the copy.

// For tentative adds internally this is to support CostToAddSpecs(). Each Spec
// within the set must be tentatively added to a block to calculate the
// potential impact on adding an additional spec. But the series of Specs added
// needs to be reversable, in that after the total cost is calculated the
// Strategy may decide to not add that list of Specs and so they need to be
// removed. This means that Block needs AppendTentative(), ResetTentative(), and
// CommitTentative() functions, and modifications to other functions like
// final_state(), range(), and CostToAppend() to support providing the
// information also including the tentative state.
class Schedule {
 public:
  Schedule();
  Schedule(const Schedule& schedule);
  ~Schedule();

  // Makes a copy-on-write copy of this schedule.
  std::unique_ptr<Schedule> MakeLazyCopy();

  // Returns the cost in color clocks of adding this spec. If it cannot add
  // this spec it will return kInfinity.
  uint64 CostToAddSpec(const Spec& spec);

  // To avoid the _factorial_ explosion of possibilities of order adding the
  // Schedule will add specs in the order provded by the list.
  uint64 CostToAddSpecs(const std::vector<Spec>* specs);

  uint64 AddSpec(const Spec& spec);
  uint64 AddSpecs(const std::vector<Spec>* specs);

  std::string Assemble() const;

  const State* initial_state() const { return (*states_.begin()).get(); }
  const State* final_state() const { return (*states_.rbegin()).get(); }

 private:
  typedef std::vector<std::unique_ptr<State>> States;
  typedef std::vector<std::unique_ptr<Block>> Blocks;

  uint8 initial_tia_values_[TIA::TIA_COUNT];
  // |states_| and |blocks_| are expected to be interleaved, with the State
  // objects bookending the Blocks on both sides.
  States states_;
  Blocks blocks_;
};

}  // namespace vcsmc

#endif  // SRC_SCHEDULE_H_
