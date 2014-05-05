#ifndef SRC_BACKGROUND_COLOR_STRATEGY_H_
#define SRC_BACKGROUND_COLOR_STRATEGY_H_

#include "strategy.h"

namespace vcsmc {

// Attempts to fit the provided ScanLine by only changing the background color.
class BackgroundColorStrategy : public Strategy {
 public:
  virtual std::unique_ptr<ScanLine> Fit(ColuStrip* target_strip,
                                        State* entry_state) override;
};

}  // namespace vcsmc

#endif  // SRC_BACKGROUND_COLOR_STRATEGY_H_
