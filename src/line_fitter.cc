#include "line_fitter.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include "line_kernel.h"
#include "opcode.h"
#include "random.h"

namespace vcsmc {

// When evaluating a Line program, how many scanlines should the program be
// scored against? Should be at least 1, to evaluate the line the program is
// running on, but can be more.
const uint32 kLinesToScore = 2;
static const uint32 kGenerationSize = 500;
static const uint32 kBoutSize = kGenerationSize / 10;
static const uint32 kMaxGenerations = 25;
static const float kMaxError = 100.0f;

LineFitter::LineFitter() {
}

LineFitter::~LineFitter() {
}

float LineFitter::Fit(Random* random, const uint8* half_colus, uint32 scan_line,
    const State* entry_state) {
  std::vector<std::unique_ptr<LineKernel>> population;
  population.reserve(2 * kGenerationSize);

  // Generate initial generation of programs.
  for (uint32 i = 0; i < kGenerationSize; ++i) {
    std::unique_ptr<LineKernel> lk(new LineKernel());
    lk->Randomize(random);
    population.push_back(std::move(lk));
  }

  uint32 best = SimulatePopulation(population, half_colus, scan_line,
    entry_state);
  uint32 generation_count = 0;
  while (generation_count < kMaxGenerations &&
      population[best]->sim_error() > kMaxError) {
    // Each member of current population generates one offspring by cloning
    // followed by mutation.
    for (uint32 i = 0; i < kGenerationSize; ++i) {
      std::unique_ptr<LineKernel> lk = population[i]->Clone();
      lk->Mutate(random);
      population.push_back(std::move(lk));
    }

    // Simulate new offspring.
    SimulatePopulation(population, half_colus, scan_line, entry_state);

    // Compete to survive!
    for (uint32 i = 0; i < population.size(); ++i) {
      for (uint32 j = 0; j < kBoutSize; ++j) {
        population[i]->Compete(
            population[random->Next() % population.size()].get());
      }
    }

    // Sort by victories greatest to least, then remove the lower half of the
    // population.
    std::sort(population.begin(),
              population.end(),
              &LineFitter::CompareKernels);
    population.erase(population.begin() + kGenerationSize, population.end());

    best = 0;
    uint32 worst = 0;
    float sum = population[0]->sim_error();
    for (uint32 i = 1; i < population.size(); ++i) {
      sum += population[i]->sim_error();
      if (population[i]->sim_error() < population[best]->sim_error())
        best = i;
      if (population[i]->sim_error() > population[worst]->sim_error())
        worst = i;
    }

    float mean = sum / (float)(population.size());

    printf("  gen %d, best: %f, best cycles: %d, mean: %f, worst: %f\n",
        generation_count,
        population[best]->sim_error(),
        population[best]->total_cycles(),
        mean,
        population[worst]->sim_error());
    ++generation_count;
  }

  // Save the best final fit from the vector, simulation results included.
  best_fit_.swap(population[best]);
  return best_fit_->sim_error();
}

void LineFitter::AppendBestFit(
    std::vector<std::unique_ptr<op::OpCode>>* opcodes,
    std::vector<std::unique_ptr<State>>* states) {
  assert(best_fit_);
  best_fit_->Terminate();
  best_fit_->Append(opcodes, states);
}

uint32 LineFitter::SimulatePopulation(
    const std::vector<std::unique_ptr<LineKernel>>& population,
    const uint8* half_colus,
    uint32 scan_line,
    const State* entry_state) {
  uint32 best_index = 0;
  for (uint32 i = 0; i < population.size(); ++i) {
    population[i]->Simulate(half_colus, scan_line, entry_state, kLinesToScore);
    if (population[i]->sim_error() < population[best_index]->sim_error())
      best_index = i;
  }
  return best_index;
}

// static
bool LineFitter::CompareKernels(const std::unique_ptr<LineKernel>& lk1,
                                const std::unique_ptr<LineKernel>& lk2) {
  return lk1->victories() > lk2->victories();
}

}  // namespace vcsmc
