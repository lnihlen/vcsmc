extern "C" {
#include "libz26/libz26.h"
}

#include "cuda.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include "cuda_utils.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  init_z26_global_tables();
  if (!vcsmc::InitializeCuda(false)) return -1;

  {
    int result = RUN_ALL_TESTS();
    return result;
  }
}
