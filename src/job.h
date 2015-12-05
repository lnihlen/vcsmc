#ifndef SRC_JOB_H_
#define SRC_JOB_H_

namespace vcsmc {

class Job {
 public:
   virtual void Execute() = 0;
   virtual ~Job() {}
};


}  // namespace vcsmc

#endif  // SRC_JOB_H_
