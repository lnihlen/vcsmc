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
  Schedule(const Schedule& schedule);
  const Schedule& operator=(const Schedule& schedule);
  ~Schedule();

  uint32 AddSpec(const Spec& spec);
  uint32 AddSpecs(const std::list<Spec>* specs);
  // Returns the cost in color clocks of adding this spec. If it cannot add
  // this spec it will return kInfiniteCost.
  uint32 CostToAddSpec(const Spec& spec);
  // To avoid the _factorial_ explosion of possibilities of order adding the
  // Schedule will add specs in the order provded by the list.
  uint32 CostToAddSpecs(const std::list<Spec>* specs);

  std::unique_ptr<ColuStrip> Simulate(uint32 row);

 private:
  // |states_| and |time_spans_| are expected to be interleaved, with the State
  // objects bookending the TimeSpans.
  std::list<std::unique_ptr<State>> states_;
  std::list<std::unique_ptr<Block>> blocks_;
};

}  // namespace vcsmc

#endif  // SRC_SCHEDULE_H_
