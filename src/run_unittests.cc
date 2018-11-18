extern "C" {
#include "libz26/libz26.h"
}

#include "gtest/gtest.h"

#include "cuda_utils.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  init_z26_global_tables();

  {
    int result = RUN_ALL_TESTS();
    return result;
  }
}
