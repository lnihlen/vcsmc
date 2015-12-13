#include "gtest/gtest.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  int result = RUN_ALL_TESTS();
  return result;
}
