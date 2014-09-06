#include "assembler.h"
#include "cl_device_context.h"
#include "gtest/gtest.h"

int main(int argc, char* argv[]) {
  vcsmc::CLDeviceContext::Setup();
  vcsmc::Assembler::InitAssemblerTables();

  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  int result = RUN_ALL_TESTS();

  vcsmc::CLDeviceContext::Teardown();
  return result;
}
