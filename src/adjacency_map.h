#ifndef SRC_ADJACENCY_MAP_H_
#define SRC_ADJACENCY_MAP_H_

#include <memory>

#include "types.h"

namespace vcsmc {

class BitMap;

// Given an input bitmap, produces an array of uint8s of same dimensions that
// stores at each position the number of bits that are set to 1 in the adjacent
// 8 pixels. This is useful predominantly for player fitting, as the
// AdjacencyMap becomes the input coverage map for the dynamic programming
// algorithm there.
//
// Example saliency map below:
// 0 0 0 0 1 0 1 0 1 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0 1 1 0 0 1 0 1 0 0 0 1 0 1 0 1
// 2 3 3 3 3 3 3 3 3 2 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 3 2 3 3 3 3 3 3 3 3 2 2 1 1
// And corresponding adjacency map above.
//
class AdjacencyMap {
 public:
  AdjacencyMap();
  ~AdjacencyMap();

  void Build(const BitMap* bitmap);

  const uint8* map() const { return map_.get(); }
  uint8 count_at(uint32 x, uint32 y) const { return map_[(y * width_) + x]; }
  bool line_empty(uint32 y) const { return line_empty_[y]; }

 private:
  uint32 width_;
  std::unique_ptr<uint8[]> map_;
  std::unique_ptr<bool[]> line_empty_;
};

}  // namespace vcsmc

#endif
