#include "adjacency_map.h"
#include "bit_map.h"
#include "gtest/gtest.h"
#include "types.h"

namespace vcsmc {

TEST(AdjacencyMapTest, EmptyRowIsAllZeros) {
  BitMap bitmap(33, 1);
  for (uint32 i = 0; i < 33; ++i)
    bitmap.SetBit(i, 0, false);

  AdjacencyMap m;
  m.Build(&bitmap);

  for (uint32 i = 0; i < 33; ++i)
    EXPECT_EQ(0, m.count_at(i, 0));

  EXPECT_TRUE(m.line_empty(0));
}

TEST(AdjacencyMapTest, FullRow) {
  BitMap bitmap(47, 2);
  for (uint32 i = 0; i < 47; ++i) {
    bitmap.SetBit(i, 0, false);
    bitmap.SetBit(i, 1, true);
  }

  AdjacencyMap m;
  m.Build(&bitmap);

  for (uint32 i = 0; i < 47 - 8; ++i)
    EXPECT_EQ(8, m.count_at(i, 1));

  for (uint32 i = 0; i < 8; ++i)
    EXPECT_EQ(8 - i, m.count_at(i + 47 - 8, 1));

  EXPECT_TRUE(m.line_empty(0));
  EXPECT_FALSE(m.line_empty(1));
}

TEST(AdjacencyMapTest, SingleBitCounts) {
  BitMap bitmap(64, 8);
  for (uint32 i = 0; i < 8; ++i) {
    for (uint32 j = 0; j < 64; ++j) {
      bitmap.SetBit(j, i, (j % 8) == i);
    }
  }

  AdjacencyMap m;
  m.Build(&bitmap);

  for (uint32 i = 0; i < 8; ++i) {
    for (uint32 j = 0; j < 64 - (7 - i); ++j) {
      EXPECT_EQ(1, m.count_at(j, i));
    }
    for (uint32 j = 0; j < 7 - i; ++j) {
      EXPECT_EQ(0, m.count_at(64 - (7 - i) + j, i));
    }

    EXPECT_FALSE(m.line_empty(i));
  }
}

TEST(AdjacencyMapTest, CompareToLoopCounts) {
  BitMap bitmap(16 * 8, 16);
  for (uint32 i = 0; i < 16; ++i) {
    for (uint32 j = 0; j < 16; ++j) {
      for (uint32 k = 0; k < 8; ++k) {
        bitmap.SetBit(k + (j * 8), i, ((i * 16) + j) & (1 << k));
      }
    }
  }

  AdjacencyMap m;
  m.Build(&bitmap);

  // Compare map counts to manually calculated count.
  for (uint32 i = 0; i < 16; ++i) {
    for (uint32 j = 0; j < 16 * 8; ++j) {
      uint8 count = 0;
      for (uint32 k = 0; k < 8; ++k) {
        if (j + k < 16 * 8) {
          if (bitmap.bit(j + k, i))
            ++count;
        }
      }
      EXPECT_EQ(count, m.count_at(j, i));
    }
    EXPECT_FALSE(m.line_empty(i));
  }
}

}  // namespace vcsmc
