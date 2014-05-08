#ifndef SRC_PLAYFIELD_STRATEGY_H_
#define SRC_PLAYFIELD_STRATEGY_H_

#include "strategy.h"

namespace vcsmc {

// Attempts to fit the provided ScanLine by using only the Playfield.
class PlayfieldStrategy : public Strategy {
 public:
  virtual std::unique_ptr<ScanLine> Fit(PixelStrip* target_strip,
                                        State* entry_state) override;
 private:
  // Given a PixelStrip and an offset (in pixels) calculates the error distance
  // from modeling that color as all 8 pixels.
  double EightPixelError(PixelStrip* target_strip, uint32 offset, uint8 colu);
};

}  // namespace vcsmc

#endif  // SRC_PLAYFIELD_STRATEGY_H_
