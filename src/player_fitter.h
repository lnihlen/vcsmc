#ifndef SRC_PLAYER_FITTER_H_
#define SRC_PLAYER_FITTER_H_

#include <memory>
#include <vector>

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

  // Convert optimal path as found BuildCoverageTables() into a series of
  // Specs for either player zero or player one, depending on value of
  // |is_player_one|.
  void AppendSpecs(std::vector<Spec>* specs, bool is_player_one);

  // Given that the player is positioned at x1 on a given scanline, returns
  // the number of color clocks that will have to be expended to move the
  // player to the position x2 on the next scanline.
  uint32 CostToMove(uint32 x1, uint32 x2);

 private:
  // For each position k in the adjacency array, consider each position on the
  // row above, calculating coverage totals resulting from jumping from that
  // position on the row above to the position under consideration. Example:
  // 8   8   8
  //     x
  // At position x if we consider the leftmost 8 the total coverage is x + 8,
  // since the reset could issue after the leftmost 8, but it has a somewhat
  // higher cost of 3 (3 clocks to STA RESPX) than the center choice, which
  // has cost of 0 and coverage of x + 8. The 8 on the right has a coverage
  // score of x + 0, missing out on the +8 because the STA RESPX will have
  // to occur at the position of x on the former scanline, therefore skipping
  // coverage of the right pixels, plus has a cost of 3.
  // At each position in the above row we compare coverage total and cost total
  // to the stored best value, and if better update stored best value, also
  // updating the go_to position - which stores the position of the pixel on the
  // row below that the best score is going to.
};

}  // namespace vcsmc

#endif  // SRC_PLAYER_FITTER_H_
