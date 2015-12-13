// gen - outputs a generation of

#include <gflags/gflags.h>

#include "job_queue.h"

DEFINE_int32(max_generation_number, 10000,
    "Number of generations of evolutionary programming to run.");
DEFINE_int32(generation_size, 50000,
    "Number of individuals to keep in an evolutionary programing generation.");
DEFINE_int32(worker_threads, 0,
    "Number of threads to create to work in parallel, set to 0 to pick based "
    "on hardware.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
/*
  vcsmc::JobQueue job_queue(FLAGS_worker_threads);
  map<kernel_hash, float score> score_map;
  std::vector<std::shared_ptr<vcsmc::kernel>> current_generation(
      FLAGS_generation_size);

  // Go look in log directory to see if we are resuming work from a previous
  // run, if so restore that state, otherwise generate all new stuff.

  // Generate FLAGS_generation_size number of random individuals.
  for (int i = 0; i < FLAGS_generation_size; ++i)
    job_queue.enque(new GenerateRandomIndividualJob());

  for (int i = 0; i < FLAGS_max_generation_number; ++i) {

// ** this is probably where the loop should start.

    // Discard worse half of generation.
    //

    // There is some annealing parameter we can consider which determines how
    // many mutations we make to the surviving programs. It can take as input
    // things like how rapidly we are descending in score or the relative rank
    // of the kernel being mutated, perhaps mutating the lower-ranked members
    // more aggressively. Thinking about a value between 0 and 1 where 0
    // represents a minimum of 1 change and 1 represents a total regeneration.
    float annealing = 0.0f;

    for (int j = 0; j < FLAGS_generation_size / 2; ++j) {
      job_queue.enqueue(new MutateIndividualJob());
    }

     // Score everyone in the generation.
    for (int j = 0; j < FLAGS_generation_size; ++j) {
      if (!current_generation[j]->has_score()) {
        job_queue.enque(new SimulateAndScoreOutputJob());
      }
    }

    // Wait for scoring to complete.
    job_queue.finish();

    // Hold tournament.
    for (int j = 0; j < FLAGS_generation_size; ++j) {
      job_queue.enqueue(new ResetScoreThenCompeteJob());
    }

    // Wait for tournament to complete.
    job_queue.finish();

    // Sort generation based on tournament results.
    //

    // Save generation to disk.
    for (int j = 0; j < FLAGS_generation_size; ++j) {
      job_queue.enqueue(new UpdateGenerationLoggingJob());
    }

    // As well update overall report log.

  }
*/
  return 0;
}
