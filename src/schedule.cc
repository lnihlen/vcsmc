#include "schedule.h"

namespace vcsmc {

Schedule::Schedule() {
}

Schedule::Schedule(const Schedule& schedule) {
  *this = schedule;
}

const Schedule& Schedule::operator=(const Schedule& schedule) {
  // TODO: copy lists
  return *this;
}

Schedule::~Schedule() {
}

uint32 Schedule::AddSpec(const Spec& spec) {
}

uint32 Schedule::AddSpecs(const std::list<Spec>* specs) {
}

uint32 Schedule::CostToAddSpec(const Spec& spec) {
}

uint32 Schedule::CostToAddSpecs(const std::list<Spec>* specs) {
}

std::unique_ptr<ColuStrip> Schedule::Simulate(uint32 row) {
}

}  // namespace vcsmc
