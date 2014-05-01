#ifndef SRC_BACKGROUND_COLOR_STRATEGY_H_
#define SRC_BACKGROUND_COLOR_STRATEGY_H_

#include "strategy.h"

namespace vcsmc {

// Attempts to fit the provided ScanLine by only changing the background color.
class BackgroundColorStrategy : public Strategy {
 public:
  BackgroundColorStrategy(const ColuStrip& target_strip,
                          const State& entry_state);

  virtual ScanLine Fit() override;
};

}  // namespace vcsmc

#endif  // SRC_BACKGROUND_COLOR_STRATEGY_H_
