#ifndef SRC_STRATEGY_H_
#define SRC_STRATEGY_H_

#include "colu_strip.h"
#include "scan_line.h"
#include "state.h"

namespace vcsmc {

// A Strategy defines an approach to generating ScanLines. It has a target line
// of colors and attempts to generate a ScanLine with minimum error and length
// that approximates that ScanLine. This defines an abstract base interface for
// that behavior.
class Strategy {
 public:
  // Attempts to fit the colustrip, returns a new ScanLine representing best
  // fit.
  virtual std::unique_ptr<ScanLine> Fit(
      const std::unique_ptr<ColuStrip>& target_strip,
      const std::unique_ptr<State>& entry_state) = 0;
};

}  // namespace vcsmc

#endif  // SRC_STRATEGY_H_
