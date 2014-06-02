#include "schedule.h"

#include <cassert>

namespace vcsmc {

Schedule::Schedule() {
  states_.push_back(std::unique_ptr<State>(new State));
}

Schedule::Schedule(const Schedule& schedule) {
  *this = schedule;
}

const Schedule& Schedule::operator=(const Schedule& schedule) {
  // TODO: copy vectors
  return *this;
}

Schedule::~Schedule() {
}

uint32 Schedule::AddSpec(const Spec& spec) {
}

uint32 Schedule::AddSpecs(const std::list<Spec>* specs) {
}

uint32 Schedule::CostToAddSpec(const Spec& spec) {
  assert(states_.size() > 0);

  // Iterate backward through states, looking for a State that returns nonzero
  // scheduling time.
  Blocks::reverse_iterator block_it = blocks_.rbegin();
  States::reverse_iterator state_it = states_.rbegin();

  // ** This needs to happen in the latest state before the deadline.
  // Is it possible that this state already has what we want? If so we need
  // to add nothing, this state is for free.
  if ((*state_it)->tia(spec.tia()) == spec.value())
    return 0;

  for (; state_it != states_.rend(); state_it++) {
    // If this state ends before this spec can be scheduled we need to go
    // further back.
    if ((*state_it)->range().end_time() < spec.range().start_time())
      continue;

    // If this state begins before this spec can be scheduled then we are at
    // the correct state to schedule something.
    if ((*state_it)->range().start_time() < spec.range().start_time())
      break;

    uint32 state_clock = (*state_it)->EarliestTimeAfter(spec);
    if (state_clock == kInfinity)
      return kInfinity;

    // Nonzero return from EarliestTimeAfter(spec) means that we have found the
    // state after which we want to schedule an operation.
    if (state_clock > 0)
      break;

    if (blocks_it != blocks_.rend())
      blocks_it++;
  }

  // We should have ended on a valid state, this is an odd error condition. Need
  // to give more thought on what it means.
  if (state_it == states_.rend()) {
    assert(false);
    return kInfinity;
  }

  // Now we advance forward in |states_| and |blocks_| until we find room for
  // the Spec.
  for (; state_it >= states_.rbegin(); state_it--) {
    // If we are no longer within the |spec| time range than we can't
    // accommodate this spec.
    if (Range::IntersectRanges((*state_it)->range(), spec.range()).IsEmpty())
      return kInfinity;

    // Are we pointing at a block? Can we append to this block?
    if (blocks_it != blocks_.rend()) {
      uint32 block_append_cost = (*blocks_it)->CostToAppend(spec);
      uint32 block_end_time =
          (*blocks_it)->range().end_time() + block_append_cost;
      Blocks::reverse_iterator next_block =
          blocks_it == blocks_.rbegin() ? blocks_.rend() : blocks_it - 1;

      // Would adding to this block allow the spec to occur within time?
      if (spec.range().Contains(block_end_time)) {
        // Would adding to this block cause us to run into the next block, or
        // create a gap smaller than the kMinimumIdleTime?
        if (next_block != blocks_.rend()) {
          if ((*next_block)->range().start_time() - kMinimumIdleTime <
              block_end_time) {
            blocks_it = next_block;
            continue;
          } else if ((*next_block)->range().start_time() == block_end_time) {
            // If these times align perfectly than we can merge the two blocks.
            return block_append_cost;
          }
        }
      } else {
        // An append to the current block is not within the |spec| time range.
        // Either the block ends too early, or the block ends too late. If
        // too early we can start a new block.
      }
    }
  }
}

uint32 Schedule::CostToAddSpecs(const std::list<Spec>* specs) {
}

std::unique_ptr<ColuStrip> Schedule::Simulate(uint32 row) {
}


}  // namespace vcsmc
