#ifndef SRC_DO_NOTHING_STRATEGY_H_
#define SRC_DO_NOTHING_STRATEGY_H_

#include "strategy.h"

namespace vcsmc {

class DoNothingStrategy : public Strategy {
 public:
  virtual std::unique_ptr<ScanLine> Fit(
    const std::unique_ptr<ColuStrip>& target_strip,
    const std::unique_ptr<State>& entry_state) override;
};

}  // namespace vcsmc

#endif  // SRC_DO_NOTHING_STRATEGY_H_
