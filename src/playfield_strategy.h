#ifndef SRC_PLAYFIELD_STRATEGY_H_
#define SRC_PLAYFIELD_STRATEGY_H_

#include "strategy.h"

namespace vcsmc {

// Attempts to fit the provided PixelStrip by using only the Playfield.
class PlayfieldStrategy : public Strategy {
 public:
  virtual std::unique_ptr<Schedule> Fit(
      const PixelStrip* target_strip,
      const Schedule* starting_schedule) override;
};

}  // namespace vcsmc

#endif  // SRC_PLAYFIELD_STRATEGY_H_
