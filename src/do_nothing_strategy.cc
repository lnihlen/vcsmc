#include "do_nothing_strategy.h"

#include <cassert>

#include "schedule.h"

namespace vcsmc {

std::unique_ptr<Schedule> DoNothingStrategy::Fit(
    const PixelStrip* target_strip, const Schedule* starting_schedule) {
  // Doing nothing means the schedule will be the same at exit that it was at
  // entry.
  return std::unique_ptr<Schedule>(new Schedule(*starting_schedule));
}

}  // namespace vcsmc
