#include "adjacency_map.h"

#include <cassert>

#include "bit_map.h"

namespace vcsmc {

AdjacencyMap::AdjacencyMap() {
}

AdjacencyMap::~AdjacencyMap() {
}

void AdjacencyMap::Build(const BitMap* bitmap) {
  width_ = bitmap->width();
  assert(width_ >= 16);
  uint32 bm_size = width_ * bitmap->height();
  map_.reset(new uint8[bm_size]);
  line_empty_.reset(new bool[bitmap->height()]);

  uint8* map_ptr = map_.get();
  for (uint32 i = 0; i < bitmap->height(); ++i) {
    bool line_empty = true;
    uint8 counter = 0;
    // Add up first 8 bits, as this is the first value in the map for this row.
    for (uint32 j = 0; j < 8; ++j) {
      if (bitmap->bit(j, i)) {
        ++counter;
        line_empty = false;
      }
    }

    *map_ptr = counter;
    ++map_ptr;

    // Now we move from the 8th pixel to the end. |j| here identifies the right
    // edge of the 8-pixel window we are summing pixel counts for. Note that
    // |map_ptr| lags j by 8 pixels in the adjacency map.
    for (uint32 j = 8; j < width_; ++j) {
      if (bitmap->bit(j, i)) {
        ++counter;
        line_empty = false;
      }
      if (bitmap->bit(j - 8, i))
        --counter;
      *map_ptr = counter;
      ++map_ptr;
    }

    // Last 8 pixels just have decreasing amounts of overlap.
    // TODO: consider accounting for wraparound to next line, low priority
    // because these rightward positions tend to be very high cost to move
    // out of.
    for (uint32 j = 0; j < 7; ++j) {
      if (bitmap->bit(width_ - 8 + j, i))
        --counter;
      *map_ptr = counter;
      ++map_ptr;
    }

    line_empty_[i] = line_empty;
  }
}

}  // namespace vcsmc
