#include "range.h"

#include "gtest/gtest.h"

TEST(RangeTest, DefaultRangeIsEmpty) {
  vcsmc::Range range;
  EXPECT_TRUE(range.IsEmpty());
}

TEST(RangeTest, DefaultRangeHasNoDuration) {
  vcsmc::Range range;
  EXPECT_EQ(0U, range.Duration());
}

TEST(RangeTest, ArgumentConstructorCopiesValues) {
  vcsmc::Range range(0, 10);
  EXPECT_EQ(0U, range.start_time());
  EXPECT_EQ(10U, range.end_time());
}

TEST(RangeTest, CopyConstructorCopiesValues) {
  vcsmc::Range range1(2, 15);
  vcsmc::Range range_copy(range1);
  EXPECT_EQ(range1.start_time(), range_copy.start_time());
  EXPECT_EQ(range1.end_time(), range_copy.end_time());
}

TEST(RangeTest, OperatorEqualCopiesValues) {
  vcsmc::Range range1(5, 45);
  vcsmc::Range range_copy = range1;
  EXPECT_EQ(range1.start_time(), range_copy.start_time());
  EXPECT_EQ(range1.end_time(), range_copy.end_time());
}

TEST(RangeTest, ContainsBeforeRange) {
  vcsmc::Range range(2, 25);
  EXPECT_FALSE(range.Contains(1));
}

TEST(RangeTest, ContainsAfterRange) {
  vcsmc::Range range(1, 100);
  EXPECT_FALSE(range.Contains(200));
}

TEST(RangeTest, ContainsRightAtStart) {
  vcsmc::Range range(1000, 2000);
  EXPECT_TRUE(range.Contains(1000));
}

TEST(RangeTest, ContainsRightAtEnd) {
  vcsmc::Range range(500, 999);
  EXPECT_FALSE(range.Contains(999));
}

TEST(RangeTest, ContainsRightInMiddle) {
  vcsmc::Range range(700, 1400);
  EXPECT_TRUE(range.Contains(1050));
}

TEST(RangeTest, DurationZeroStartTime) {
  vcsmc::Range range(0, 500);
  EXPECT_EQ(range.end_time(), range.Duration());
}

TEST(RangeTest, DurationNonZeroStartTime) {
  vcsmc::Range range(250, 475);
  EXPECT_EQ(225U, range.Duration());
}

TEST(RangeTest, IsEmptyZeroStartTime) {
  vcsmc::Range range(0, 0);
  EXPECT_TRUE(range.IsEmpty());

  vcsmc::Range range_non_empty(0, 1);
  EXPECT_FALSE(range_non_empty.IsEmpty());
}

TEST(RangeTest, IsEmptyNonZeroStartTime) {
  vcsmc::Range range(10, 10);
  EXPECT_TRUE(range.IsEmpty());

  vcsmc::Range range_non_empty(10, 11);
  EXPECT_FALSE(range_non_empty.IsEmpty());
}

TEST(RangeDeathTest, IllPosedConstructionAsserts) {
  EXPECT_DEATH(vcsmc::Range range(5, 0), "end_time_ >= start_time_");
}

TEST(RangeDeathTest, BadStartTimeAsserts) {
  vcsmc::Range range(1000, 2000);
  EXPECT_DEATH(range.set_start_time(2001), "end_time_ >= start_time");
}

TEST(RangeDeathTest, BadEndTimeAsserts) {
  vcsmc::Range range(20, 35);
  EXPECT_DEATH(range.set_end_time(10), "end_time >= start_time_");
}
