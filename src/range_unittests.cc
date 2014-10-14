#include "constants.h"
#include "gtest/gtest.h"
#include "range.h"

namespace vcsmc {

TEST(RangeTest, DefaultRangeIsEmpty) {
  Range range;
  EXPECT_TRUE(range.IsEmpty());
}

TEST(RangeTest, DefaultRangeHasNoDuration) {
  Range range;
  EXPECT_EQ(0U, range.Duration());
}

TEST(RangeTest, ArgumentConstructorCopiesValues) {
  Range range(0, 10);
  EXPECT_EQ(0U, range.start_time());
  EXPECT_EQ(10U, range.end_time());
}

TEST(RangeTest, CopyConstructorCopiesValues) {
  Range range1(2, 15);
  Range range_copy(range1);
  EXPECT_EQ(range1.start_time(), range_copy.start_time());
  EXPECT_EQ(range1.end_time(), range_copy.end_time());
}

TEST(RangeTest, OperatorEqualCopiesValues) {
  Range range1(5, 45);
  Range range_copy = range1;
  EXPECT_EQ(range1.start_time(), range_copy.start_time());
  EXPECT_EQ(range1.end_time(), range_copy.end_time());
}

TEST(RangeTest, ComparisonRangesEqual) {
  Range range(7, 13);
  Range range2(7, 13);
  EXPECT_TRUE(range == range2);
  EXPECT_FALSE(range != range2);
}

TEST(RangeTest, ComparisonStartTimesDifferent) {
  Range range(25, 26);
  Range range2(13, 26);
  EXPECT_FALSE(range == range2);
  EXPECT_TRUE(range != range2);
}

TEST(RangeTest, ComparisonEndTimesDifferent) {
  Range range(75, 99);
  Range range2(75, 275);
  EXPECT_FALSE(range == range2);
  EXPECT_TRUE(range != range2);
}

TEST(RangeTest, ComparisonBothTimesDifferent) {
  Range range(173, 999);
  Range range2(411, 193574);
  EXPECT_FALSE(range == range2);
  EXPECT_TRUE(range != range2);
}

TEST(RangeTest, ContainsBeforeRange) {
  Range range(2, 25);
  EXPECT_FALSE(range.Contains(1));
}

TEST(RangeTest, ContainsAfterRange) {
  Range range(1, 100);
  EXPECT_FALSE(range.Contains(200));
}

TEST(RangeTest, ContainsRightAtStart) {
  Range range(1000, 2000);
  EXPECT_TRUE(range.Contains(1000));
}

TEST(RangeTest, ContainsRightAtEnd) {
  Range range(500, 999);
  EXPECT_FALSE(range.Contains(999));
}

TEST(RangeTest, ContainsRightInMiddle) {
  Range range(700, 1400);
  EXPECT_TRUE(range.Contains(1050));
}

TEST(RangeTest, DurationZeroStartTime) {
  Range range(0, 500);
  EXPECT_EQ(range.end_time(), range.Duration());
}

TEST(RangeTest, DurationNonZeroStartTime) {
  Range range(250, 475);
  EXPECT_EQ(225U, range.Duration());
}

TEST(RangeTest, IsEmptyZeroStartTime) {
  Range range(0, 0);
  EXPECT_TRUE(range.IsEmpty());

  Range range_non_empty(0, 1);
  EXPECT_FALSE(range_non_empty.IsEmpty());
}

TEST(RangeTest, IsEmptyNonZeroStartTime) {
  Range range(10, 10);
  EXPECT_TRUE(range.IsEmpty());

  Range range_non_empty(10, 11);
  EXPECT_FALSE(range_non_empty.IsEmpty());
}

TEST(RangeTest, IntersectRangeBefore) {
  Range range(1, 25);
  Range range2(25, 30);
  Range intersect(Range::IntersectRanges(range, range2));
  EXPECT_TRUE(intersect.IsEmpty());
}

TEST(RangeTest, IntersectRangeAfter) {
  Range range(200, 284);
  Range range2(20, 100);
  Range intersect(Range::IntersectRanges(range, range2));
  EXPECT_TRUE(intersect.IsEmpty());
}

TEST(RangeTest, IntersectRangeLarger) {
  Range range(10, 10000);
  Range range2(50, 60);
  Range intersect(Range::IntersectRanges(range, range2));
  EXPECT_EQ(range2, intersect);
}

TEST(RangeTest, IntersectRangeIdentical) {
  Range range(75, 99);
  Range range2(range);
  Range intersect(Range::IntersectRanges(range, range2));
  EXPECT_EQ(range, intersect);
  EXPECT_EQ(range2, intersect);
}

TEST(RangeTest, IntersectRangeSmaller) {
  Range range(75, 99);
  Range range2(50, 999);
  Range intersect(Range::IntersectRanges(range, range2));
  EXPECT_EQ(range, intersect);
}

TEST(RangeTest, IntersectRangeEmptyRange) {
  Range empty_range(75, 75);
  Range range(42, 1336);
  Range intersect = Range::IntersectRanges(empty_range, range);
  EXPECT_TRUE(intersect.IsEmpty());
}

TEST(RangeTest, SerializeToBufferEmptyRange) {
  uint8 buffer[8];
  Range range;
  EXPECT_EQ(8, range.Serialize(buffer));
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(0, buffer[i]);
}

TEST(RangeTest, SerializeToBufferInfiniteRange) {
  uint8 buffer[8];
  Range range(0, kInfinity);
  EXPECT_EQ(8, range.Serialize(buffer));
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(0, buffer[i]);
  for (int i = 4; i < 8; ++i)
    EXPECT_EQ(0xff, buffer[i]);
}

TEST(RangeTest, SerializeToBufferNonZeroLittleEndian) {
  uint8 buffer[8];
  Range range(0x01234567, 0x89abcdef);
  EXPECT_EQ(8, range.Serialize(buffer));
  uint8 counter = 0x67;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(counter, buffer[i]);
    counter -= 0x22;
  }
  counter = 0xef;
  for (int i = 4; i < 8; ++i) {
    EXPECT_EQ(counter, buffer[i]);
    counter -= 0x22;
  }
}

TEST(RangeTest, IntersectRangeZero) {
  Range empty_range;
  Range range(0, 1777);
  Range intersect = Range::IntersectRanges(empty_range, range);
  EXPECT_TRUE(intersect.IsEmpty());
}

TEST(RangeDeathTest, IllPosedConstructionAsserts) {
  EXPECT_DEATH(Range range(5, 0), "end_time_ >= start_time_");
}

TEST(RangeDeathTest, BadStartTimeAsserts) {
  Range range(1000, 2000);
  EXPECT_DEATH(range.set_start_time(2001), "end_time_ >= start_time");
}

TEST(RangeDeathTest, BadEndTimeAsserts) {
  Range range(20, 35);
  EXPECT_DEATH(range.set_end_time(10), "end_time >= start_time_");
}

}  // namespace vcsmc
