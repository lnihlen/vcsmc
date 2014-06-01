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

    // Nonzero return from EarliestTimeAfter(spec)
    if (state_clock > 0)
      break;

    if (blocks_it != blocks_.rend())
      blocsks_it++;
  }

  // We should have ended on a valid state, this is an odd error condition. Need
  // to give more thought on what it means.
  if (state_it == states_.rend()) {
    assert(false);
    return kInfinity;
  }

  // Is it possible that this state already has what we want?
  if ((*state_it)->tia(spec.tia()) == spec.value())
    return 0;

  // Are we pointing at a block? Can we append to this block?
  if (blocks_it != blocks_.rend()) {
    (*blocks_it)->CostToAppend()
  }
}

uint32 Schedule::CostToAddSpecs(const std::list<Spec>* specs) {
}

std::unique_ptr<ColuStrip> Schedule::Simulate(uint32 row) {
}


}  // namespace vcsmc
