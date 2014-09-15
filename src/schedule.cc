#include "schedule.h"

#include <cassert>

#include "block.h"
#include "colu_strip.h"
#include "opcode.h"
#include "spec.h"
#include "state.h"

namespace vcsmc {

Schedule::Schedule() {
  std::memset(initial_tia_values_, 0, sizeof(initial_tia_values_));
  states_.push_back(std::unique_ptr<State>(new State(initial_tia_values_)));
}

Schedule::Schedule(const Schedule& schedule) {
  // TODO: copy ctor
}

Schedule::~Schedule() {
}

// Revisit idea of just keeping Specs, and "scheduling" consists of the act of
// sorting the Specs, first time by end_times(). Then possibly sort by
// earliest_time_after?

// BLOCK WELDING/REPACKING
// Key idea: Blocks start out with all 3 Registers unknown.
// Block should keep a list of Specs as they are appended to the Block.

// Plans for AddSpec:
// a) Iterate backwards through |states_| looking for a state with range that
// intersects the |spec|. One such state should always exist, because the
// schedule states always cover the entire range of the program. For each state
// that occurs before intersection it must return 0 for EarliestTimeAfter() or
// this |spec| cannot be scheduled?  May require a modification to our state
// logic to temporarily extend the end_time of the |spec| to the end time of
// the state in question or some other fix to allow determination of if
// scheduling is permitted before? Or is it better to require that the end_times
// of the |spec| being added must always increase monotonically?
//
// BLOCK FINDING
//
// b) Now that we have a state in range of the Spec, or we are always using
// the final state, we call EarliestTimeAfter() on it to see if we can schedule
// it before the final state. We use the end_time() of the Spec for the
// EarliestTimeAfter call. Several possiblities exist:
// b.1) final_state returns 0. This means we can schedule before. We iterate
// back through blocks and states to determine what Block, if any, we can append
// to.
// b.2) final_state returns non-zero value.
uint64 Schedule::AddSpec(const Spec& spec) {
  // Iterate backwards through states looking for a state with range that
  // intersects the |spec|.
  // int state_index = states_.size() - 1;
  return kInfinity;
}

uint64 Schedule::AddSpecs(const std::vector<Spec>* specs) {
  return kInfinity;
}

uint64 Schedule::CostToAddSpec(const Spec& spec) {
  assert(states_.size() > 0);

  // Iterate backward through states, looking for a State that returns nonzero
  // scheduling time.
  int block_index = blocks_.size() - 1;
  int state_index = states_.size() - 1;

  // ** This needs to happen in the latest state before the deadline.
  // Is it possible that this state already has what we want? If so we need
  // to add nothing, this state is for free.
  if (states_[state_index]->tia(spec.tia()) == spec.value())
    return 0;

  for (; state_index >= 0; --state_index) {
    const State* state = states_[state_index].get();
    // If this state ends before this spec can be scheduled we need to go
    // further back.
    if (state->range().end_time() < spec.range().start_time())
      continue;

    // If this state begins before this spec can be scheduled then we are at
    // the correct state to schedule something.
    if (state->range().start_time() < spec.range().start_time())
      break;

    uint64 state_clock = state->EarliestTimeAfter(spec);
    if (state_clock == kInfinity)
      return kInfinity;

    // Nonzero return from EarliestTimeAfter(spec) means that we have found the
    // state after which we want to schedule an operation.
    if (state_clock > 0)
      break;

    if (block_index >= 0)
      --block_index;
  }

  // We should have ended on a valid state, this is an odd error condition. Need
  // to give more thought on what it means.
  if (state_index == -1) {
    assert(false);
    return kInfinity;
  }

  // Now we advance forward in |states_| and |blocks_| until we find room for
  // the Spec.
  for (; state_index < states_.size(); ++state_index) {
    const State* state = states_[state_index].get();
    // If we are no longer within the |spec| time range than we can't
    // accommodate this spec.
    if (Range::IntersectRanges(state->range(), spec.range()).IsEmpty())
      return kInfinity;

    // Are we pointing at a block? Can we append to this block?
    if (block_index < blocks_.size() && block_index > 0) {
      const Block* block = blocks_[block_index].get();
      uint32 block_append_cost = block->ClocksToAppend(spec);
      uint32 block_end_time = block->range().end_time() + block_append_cost;

      // Would adding to this block allow the spec to occur within time?
      if (spec.range().Contains(block_end_time)) {
        // Would adding to this block cause us to run into the next block, or
        // create a gap smaller than the kMinimumIdleTime?
        if (block_index + 1 < blocks_.size()) {
          const Block* next_block = blocks_[block_index + 1].get();
          if (next_block->range().start_time() - kMinimumIdleTime <
              block_end_time) {
            ++block_index;
            continue;
          } else if (next_block->range().start_time() == block_end_time) {
            // If these times align perfectly than we can merge the two blocks.
            return block_append_cost;
          }
        }
      } else {
        // An append to the current block is not within the |spec| time range.
        // Either the block ends too early, or the block ends too late. If
        // too early we can start a new block.
        return kInfinity;
      }
    }
  }

  return kInfinity;
}

uint64 Schedule::CostToAddSpecs(const std::vector<Spec>* specs) {
  return kInfinity;
}

std::unique_ptr<ColuStrip> Schedule::Simulate(uint64 row) {
  return std::unique_ptr<ColuStrip>();
}

std::string Schedule::Assemble() const {
  return std::string();
}

}  // namespace vcsmc
