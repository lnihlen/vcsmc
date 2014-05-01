#ifndef SRC_SWITCHES_H_
#define SRC_SWITCHES_H_

#include <string>

namespace vcsmc {

// Singleton class to report command line switches to the picc program, and to
// support defaults living all in one place.
class Switches {
 public:
  // Parse command-arguments. Returns false on failure to parse. Also
  // initializes the Switches singleton for future queries.
  static bool Parse(int argc, char* argv[]);
  // Call on program termination, deletes the singleton.
  static void Teardown();

  static const std::string& input_file();
  static const std::string& output_file();

 private:
  static Switches* instance_;
  std::string input_file_;
  std::string output_file_;
};

}

#endif  // SRC_SWITCHES_H_
