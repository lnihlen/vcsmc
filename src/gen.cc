// gen - outputs a generation of

#include <array>
#include <gflags/gflags.h>
#include <stdlib.h>
#include <random>

#include "job_queue.h"
#include "kernel.h"
#include "spec.h"

DEFINE_int32(max_generation_number, 10000,
    "Number of generations of evolutionary programming to run.");
DEFINE_int32(generation_size, 50000,
    "Number of individuals to keep in an evolutionary programing generation.");
DEFINE_int32(worker_threads, 0,
    "Number of threads to create to work in parallel, set to 0 to pick based "
    "on hardware.");

DEFINE_string(spec_list_file, "", "Path to spec list yaml file.");
DEFINE_string(random_seed, "",
    "Hex string of seed for pseudorandom seed generation. If not provided one "
    "will be generated from hardware random device.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  vcsmc::SpecList spec_list = vcsmc::ParseSpecListFile(FLAGS_spec_list_file);
  if (!spec_list) {
    fprintf(stderr, "Error parsing spec list file %s.\n",
        FLAGS_spec_list_file.c_str());
    return -1;
  }

  std::array<uint32, vcsmc::kSeedSizeWords> seed_array;
  if (FLAGS_random_seed.size() == vcsmc::kSeedSizeWords * 8) {
    for (size_t i = 0; i < vcsmc::kSeedSizeWords; ++i) {
      std::string word = FLAGS_random_seed.substr(i * 8, 8);
      seed_array[i] = strtoul(word.c_str(), nullptr, 16);
    }
  } else {
    std::random_device urandom;
    for (size_t i = 0; i < vcsmc::kSeedSizeWords; ++i)
      seed_array[i] = urandom();
  }
  printf("seed: ");
  for (size_t i = 0; i < vcsmc::kSeedSizeWords; ++i)
    printf("%08x", seed_array[i]);
  printf("\n");

  std::seed_seq master_seed(seed_array.begin(), seed_array.end());
  std::default_random_engine seed_engine(master_seed);

  vcsmc::JobQueue job_queue(FLAGS_worker_threads);
  std::vector<std::shared_ptr<vcsmc::Kernel>> current_generation(
      FLAGS_generation_size);

  // TODO: Go look in log directory to see if we are resuming work from a
  // previous run, if so restore that state, otherwise generate all new stuff.

  // Generate FLAGS_generation_size number of random individuals.
  for (int i = 0; i < FLAGS_generation_size; ++i) {
    std::array<uint32, vcsmc::kSeedSizeWords> seed;
    for (size_t j = 0; j < vcsmc::kSeedSizeWords; ++j)
      seed[j] = seed_engine();
    std::seed_seq kernel_seed(seed.begin(), seed.end());
    current_generation[i].reset(new vcsmc::Kernel(kernel_seed));
    job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
        new vcsmc::Kernel::GenerateRandomKernelJob(
            current_generation[i], spec_list)));
  }

  for (int i = 0; i < FLAGS_max_generation_number; ++i) {
    /*
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
    */
  }

  return 0;
}
