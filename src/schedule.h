#ifndef SRC_SCHEDULE_H_
#define SRC_SCHEDULE_H_

#include <list>
#include <memory>

namespace vcsmc {

class Spec;
class State;
class TimeSpan;

class Schedule {
 public:
  Schedule();
  Schedule(const Schedule& Schedule);
  const Schedule& operator=(const Schedule& schedule);
  ~Schedule();

  bool AddSpec(const Spec& spec);
  // Returns true if Spec can be scheduled, false if not.
  bool CanAddSpec(const Spec& spec)

  std::unique_ptr<ColuStrip> Simulate(uint32 row);

 private:
  // |states_| and |time_spans_| are expected to be interleaved, with the State
  // objects bookending the TimeSpans.
  std::list<std::unique_ptr<State>> states_;
  std::list<std::unique_ptr<TimeSpan>> time_spans_;
};

}  // namespace vcsmc

#endif  // SRC_SCHEDULE_H_
