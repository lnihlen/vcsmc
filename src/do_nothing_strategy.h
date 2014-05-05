#ifndef SRC_DO_NOTHING_STRATEGY_H_
#define SRC_DO_NOTHING_STRATEGY_H_

#include "strategy.h"

namespace vcsmc {

class DoNothingStrategy : public Strategy {
 public:
  virtual std::unique_ptr<ScanLine> Fit(ColuStrip* target_strip,
                                        State* entry_state) override;
};

}  // namespace vcsmc

#endif  // SRC_DO_NOTHING_STRATEGY_H_
