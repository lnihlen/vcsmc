#ifndef SRC_PLAYER_FITTER_H_
#define SRC_PLAYER_FITTER_H_

#include <memory>
#include <vector>

#include "constants.h"
#include "types.h"

namespace vcsmc {

class BitMap;
class Spec;

class PlayerFitter {
 public:
  PlayerFitter();
  ~PlayerFitter();

  // Given an input bitmap of desired coverage by player graphics this function
  // will compute local coverage maxima on each line, and then build an
  // optimum coverage table and path steps through the table. If |favor_right|
  // is true the fitter will try to pick the rightmost option for fitting in
  // the event of coverage and cost ties, and if false it will try to pick the
  // leftmost option. This allows for some heuristic attempt to keep the
  // player objects separate from each other.
  void FindOptimumPath(const BitMap* bitmap, bool favor_right);

  // After a call to FindOptimumPath() we can build a bitmap describing the
  // coverage of this player fit.
  std::unique_ptr<BitMap> MakeCoverageMap() const;

  // Convert optimal path as found BuildCoverageTables() into a series of
  // Specs for either player zero or player one, depending on value of
  // |is_player_one|.
  void AppendSpecs(std::vector<Spec>* specs, bool is_player_one) const;

  // Returns true if the line y [0, kFrameHeightPixels) has no player graphics
  // scheduled for it.
  bool IsLineEmpty(uint32 y) const { return row_masks_[y] == 0; }

  uint32 row_offset(uint32 y) const { return row_offsets_[y]; }
  uint8 row_mask(uint32 y) const { return row_masks_[y]; }

 private:
  // For each row [0, kFrameHeightPixels] provides the position in pixels that
  // the player graphics should be reset for (on the prior line) to render the
  // player graphics for.
  std::unique_ptr<uint32[]> row_offsets_;

  // For each row [0, kFrameHeightPixels] provides the mask of pixels to include
  // in the player graphics.
  std::unique_ptr<uint8[]> row_masks_;
};

}  // namespace vcsmc

#endif  // SRC_PLAYER_FITTER_H_
