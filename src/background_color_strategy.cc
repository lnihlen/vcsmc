#include "background_color_strategy.h"

namespace vcsmc {

std::unique_ptr<ScanLine> BackgroundColorStrategy::Fit(
    const std::unique_ptr<ColuStrip>& target_strip,
    const std::unique_ptr<State>& entry_state) {
  // Histogram the colors in the target_strip.
  return nullptr;
}

}  // namespace vcsmc
