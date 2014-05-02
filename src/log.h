#ifndef SRC_LOG_H_
#define SRC_LOG_H_

namespace vcsmc {

// Singleton class similar to Switches, meant to be instantiated in main scope
// and then accessed globally. Because everyone hates on globals but has no
// issue if you call them a fancy name like singletons. :)
class Log {
 public:
  Log();
  ~Log();

  static bool Setup();
 private:
  static Log* instance_;
};

}  // namespace vcsmc

#endif  // SRC_LOG_H_
