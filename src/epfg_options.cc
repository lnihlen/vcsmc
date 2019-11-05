#include "epfg_options.h"

#include <gflags/gflags.h>

namespace vcsmc {

void EpfgOptions::ParseCommandLineFlags(int* argc, char** argv[]) {
  gflags::ParseCommandLineFlags(argc, argv, false);
}


}
