#ifndef SRC_EPFG_OPTIONS_H_
#define SRC_EPFG_OPTIONS_H_

namespace vcsmc {

// Can be parsed from yaml or from flags, encapsulates all the invocation
// options that are passed to epfg on startup. The yaml option allows the
// invocations of the program to be revision controlled, logged, and/or
// re-used more readily than just living in bash history.
struct EpfgOptions {
 public:
  void ParseCommandLineFlags(int* argc, char** argv[]);

};

}  // namespace vcsmc

#endif  // SRC_EPFG_OPTIONS_H_
