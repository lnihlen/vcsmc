#include "playfield_strategy.h"

#include <vector>

#include "pixel_strip.h"
#include "state.h"

namespace vcsmc {

virtual std::unique_ptr<ScanLine> Fit(PixelStrip* target_strip,
                                      State* entry_state) {
  Histogram* histo = target_strip->histo();
  assert(histo);

  uint8 colubk = histo->colu(0);
  uint8 colupf = histo->colu(1);

  std::unique_ptr<ScanLine> scan_line(new ScanLine(entry_state));

  return scan_line;
}

}  // namespace vcsmc
