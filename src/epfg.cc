// epfg - Evolutionary Programming across a Frame Group - uses Evolutionary
// Programming to optimize a population of algorithms to fit a provided
// set of target images which are presumed to be similar.

#include "epfg_options.h"

int main(int argc, char* argv[]) {
  vcsmc::EpfgOptions epfg_options;
  epfg_options.ParseCommandLineFlags(&argc, &argv);

  return 0;
}
