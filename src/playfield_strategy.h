#ifndef SRC_PLAYFIELD_STRATEGY_H_
#define SRC_PLAYFIELD_STRATEGY_H_

#include "strategy.h"

namespace vcsmc {

// Attempts to fit the provided ScanLine by using only the Playfield.
class PlayfieldStrategy : public Strategy {
 public:
  virtual std::unique_ptr<ScanLine> Fit(PixelStrip* target_strip,
                                        State* entry_state) override;
};

}  // namespace vcsmc

#endif  // SRC_PLAYFIELD_STRATEGY_H_
