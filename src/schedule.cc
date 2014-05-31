#include "schedule.h"

namespace vcsmc {

Schedule::Schedule() {

}

Schedule::Schedule(const Schedule& schedule) {
  *this = schedule;
}

const Schedule& Schedule::operator=(const Schedule& schedule) {
  // TODO: copy lists
  return *this;
}

Schedule::~Schedule() {
}

uint32 Schedule::AddSpec(const Spec& spec) {
}

uint32 Schedule::AddSpecs(const std::list<Spec>* specs) {
}

uint32 Schedule::CostToAddSpec(const Spec& spec) {
}

uint32 Schedule::CostToAddSpecs(const std::list<Spec>* specs) {
}

std::unique_ptr<ColuStrip> Schedule::Simulate(uint32 row) {
}

// Quick check - is the spec already met?
// Finding a place for a spec.
// we look at the states going in to and out of blocks. Can only append
// instructions to existing blocks or create new blocks.

// Finding the earliest state we can mutate:
// Iterate backward through states starting with the last one, and stopping
// early if we encounter the start_time of the spec range:
// (a) state returns 0 - ask the state before it.
// (b) state returns kInfinity. Well shucks. Scheduling not supported. Treat
// as failure currently.
// (c) state returns n > 0. This is now the earliest state that we can add the
// spec to.

// Now we move forward in time, starting with the earliest state, and do the
// following:

// Attempt simple register packing. If any of the three registers have the
// value we want then we can re-use and skip the load. Otherwise we pick
// register based on LRU strategy (the State can keep track of this). In order
// to avoid invalidating register packing assumptions for future blocks we will
// want to ensure that register state goes back to unknown at the start of every
// block. From this we determine the number of clocks that the spec will take
// to insert.
// Given a clock count the question is if there's room or not.

// Thinking about Specs that can themselves be scheduled. Or some kind of
// Intermediate Representation here. Or OpCodes that have a time Range
// associated with them, or a weak ref back to a Spec, so they can be moved
// around a bit if needed..

// Examine block right after state, if it exists.
// if this block ends some time on or after n. Then we can attempt to append
// instructions to this block. If no block after state, or block ends too early,
// then we create a new block starting a minimum of 2 clock cycles from last
// block (for NOP insertion) and choose to append to that.

// OK now that we have our Block, can we schedule within it? Check register
// states, for now register packing is a simple question of if any of the 3
// registers coincidentally have the value we need then we can use it and skip
// the load. There may be a different design that allows for more intelligent
// register packing but we start with this. After determining the number of
// cycles we will need to add

}  // namespace vcsmc
