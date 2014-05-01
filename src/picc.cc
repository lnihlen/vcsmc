// picc - VCS picture compiler.

#include <iostream>

#include "switches.h"

int main(int argc, char* argv[]) {
  // TODO: move Switches here so that when it goes out of scope the singleton
  // is deleted.

  if (!vcsmc::Switches::Parse(argc, argv)) {
    std::cerr << "picc usage:" << std::endl << std::endl
              << "picc <input_file> [-o output_file]" << std::endl;
    return -1;
  }

  vcsmc::Switches::Teardown();
  return 0;
}
