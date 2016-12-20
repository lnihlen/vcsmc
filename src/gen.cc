// gen - uses Evolutionary Programming to evolve a series of Kernels that can
// generate some facsimile of the supplied input image.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <gperftools/profiler.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "bit_map.h"
#include "color.h"
#include "color_table.h"
#include "image.h"
#include "image_file.h"
#include "job.h"
#include "job_queue.h"
#include "kernel.h"
#include "serialization.h"
#include "spec.h"

extern "C" {
#include <libz26/libz26.h>
}

DEFINE_bool(print_stats, false,
    "If true gen will print generation statistics to stdio periodically.");

DEFINE_int32(generation_size, 1000,
    "Number of individuals to keep in an evolutionary programming generation.");
DEFINE_int32(max_generation_number, 0,
    "Maximum number of generations of evolutionary programming to run. If zero "
    "the program will run until the minimum error percentage target is met.");
DEFINE_int32(tournament_size, 100,
    "Number of kernels each should compete against.");
DEFINE_int32(worker_threads, 0,
    "Number of threads to create to work in parallel, set to 0 to pick based "
    "on hardware.");
DEFINE_int32(save_count, 1000,
    "Number of generations to run before saving the results.");
DEFINE_int32(stagnant_generation_count, 250,
    "Number of generations without score change before randomizing entire "
    "cohort. Set to zero to disable.");
DEFINE_int32(stagnant_mutation_count, 16,
    "Number of mutations to apply to each cohort member when re-randomizing "
    "stagnant cohort.");
DEFINE_int32(stagnant_count_limit, 0,
    "If nonzero, terminate after the supplied number of randomizations.");

DEFINE_string(color_input_file, "",
    "Required - best fit color map as computed by fit.");
DEFINE_string(random_seed, "",
    "Hex string of seed for pseudorandom seed generation. If not provided one "
    "will be generated from hardware random device.");
DEFINE_string(spec_list_file, "asm/frame_spec.yaml",
    "Required - path to spec list yaml file.");
DEFINE_string(gperf_output_file, "",
    "Defining enables gperftools output to the provided profile path.");
DEFINE_string(seed_generation_file, "",
    "Optional file with saved generation data, to seed the first generation of "
    "the evolutionary programming algorithm. The generation size will be set "
    "to the number of kernels in the file. Any provided spec list will also be "
    "ignored as the kernels provide their own spec.");
DEFINE_string(generation_output_file, "",
    "Optional path to output yaml file for generation saves.");
DEFINE_string(image_output_file, "",
    "Optional file to save minimum-error simulated image to.");
DEFINE_string(global_minimum_output_file, "out/minimum.yaml",
    "Required file path to save global minimum error kernel to.");
DEFINE_string(audio_spec_list_file, "",
    "Optional file path for audio spec, will clobber any existing specs at "
    "same time in existing kernels.");
DEFINE_string(target_error, "61440.0",
    "Error at which to stop optimizing.");
DEFINE_string(append_kernel_binary, "", "Path to append kernel binary.");

class CompeteKernelJob : public vcsmc::Job {
 public:
  CompeteKernelJob(
      const vcsmc::Generation generation,
      std::shared_ptr<vcsmc::Kernel> kernel,
      std::seed_seq& seed,
      size_t tourney_size)
      : generation_(generation),
        kernel_(kernel),
        engine_(seed),
        tourney_size_(tourney_size) {}
  void Execute() override {
    kernel_->ResetVictories();
    std::uniform_int_distribution<size_t> tourney_distro(
        0, generation_->size() - 1);
    for (size_t i = 0; i < tourney_size_; ++i) {
      size_t contestant_index = tourney_distro(engine_);
      // Lower scores mean better performance.
      if (generation_->at(contestant_index)->score() >= kernel_->score())
        kernel_->AddVictory();
    }
  }
 private:
  const vcsmc::Generation generation_;
  std::shared_ptr<vcsmc::Kernel> kernel_;
  std::default_random_engine engine_;
  size_t tourney_size_;
};

void SaveState(vcsmc::Generation generation,
    std::shared_ptr<vcsmc::Kernel> global_minimum) {
  if (FLAGS_generation_output_file != "") {
    if (!vcsmc::SaveGenerationFile(generation, FLAGS_generation_output_file)) {
      fprintf(stderr, "error saving final generation file %s\n",
          FLAGS_generation_output_file.c_str());
    }
  }
  if (FLAGS_image_output_file != "") {
    global_minimum->SaveImage(FLAGS_image_output_file);
  }
  if (!vcsmc::SaveKernelToFile(global_minimum,
        FLAGS_global_minimum_output_file)) {
    fprintf(stderr, "error saving global minimum kernel %s\n",
        FLAGS_global_minimum_output_file.c_str());
  }
}

int main(int argc, char* argv[]) {
  auto program_start_time = std::chrono::high_resolution_clock::now();
  gflags::ParseCommandLineFlags(&argc, &argv, false);

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

  int col_fd = open(FLAGS_color_input_file.c_str(), O_RDONLY);
  if (col_fd < 0) {
    fprintf(stderr, "error opening color input file %s.\n",
        FLAGS_color_input_file.c_str());
    return -1;
  }
  std::unique_ptr<uint8[]> target_colors(new uint8[vcsmc::kFrameSizeBytes]);
  size_t bytes_read = read(col_fd, target_colors.get(), vcsmc::kFrameSizeBytes);
  if (bytes_read != vcsmc::kFrameSizeBytes) {
    fprintf(stderr, "file read error on color input file %s.\n",
        FLAGS_color_input_file.c_str());
    return -1;
  }
  close(col_fd);

  vcsmc::SpecList audio_spec_list;
  if (FLAGS_audio_spec_list_file != "") {
    audio_spec_list = vcsmc::ParseSpecListFile(FLAGS_audio_spec_list_file);
    if (!audio_spec_list) {
      fprintf(stderr, "error parsing audio spec list file %s.\n",
          FLAGS_audio_spec_list_file.c_str());
      return -1;
    }
  }

  std::seed_seq master_seed(seed_array.begin(), seed_array.end());
  std::default_random_engine seed_engine(master_seed);

  vcsmc::JobQueue job_queue(FLAGS_worker_threads);
  vcsmc::Generation generation;

  int generation_size = FLAGS_generation_size;
  if (FLAGS_seed_generation_file == "") {
    vcsmc::SpecList spec_list = vcsmc::ParseSpecListFile(FLAGS_spec_list_file);
    if (!spec_list) {
      fprintf(stderr, "Error parsing spec list file %s.\n",
          FLAGS_spec_list_file.c_str());
      return -1;
    }

    // Merge audio spec list into frame spec list.
    if (audio_spec_list) {
      vcsmc::SpecList merged(new std::vector<vcsmc::Spec>);
      size_t frame_spec_index = 0;
      size_t audio_spec_index = 0;
      while (audio_spec_index < audio_spec_list->size() &&
             frame_spec_index < spec_list->size()) {
        if (audio_spec_list->at(audio_spec_index).range().start_time() <
            spec_list->at(frame_spec_index).range().start_time()) {
          merged->push_back(audio_spec_list->at(audio_spec_index));
          ++audio_spec_index;
        } else {
          merged->push_back(spec_list->at(frame_spec_index));
          ++frame_spec_index;
        }
      }
      if (audio_spec_index < audio_spec_list->size()) {
        assert(frame_spec_index == spec_list->size());
        while (audio_spec_index < audio_spec_list->size()) {
          merged->push_back(audio_spec_list->at(audio_spec_index));
          ++audio_spec_index;
        }
      } else {
        assert(audio_spec_index == audio_spec_list->size());
        while (frame_spec_index < spec_list->size()) {
          merged->push_back(spec_list->at(frame_spec_index));
          ++frame_spec_index;
        }
      }
      spec_list = merged;
    }

    generation.reset(new std::vector<std::shared_ptr<vcsmc::Kernel>>);

    // Generate FLAGS_generation_size number of random individuals.
    job_queue.LockQueue();
    for (int i = 0; i < generation_size; ++i) {
      std::array<uint32, vcsmc::kSeedSizeWords> seed;
      for (size_t j = 0; j < vcsmc::kSeedSizeWords; ++j)
        seed[j] = seed_engine();
      std::seed_seq kernel_seed(seed.begin(), seed.end());
      generation->emplace_back(new vcsmc::Kernel(kernel_seed));
      job_queue.EnqueueLocked(std::unique_ptr<vcsmc::Job>(
          new vcsmc::Kernel::GenerateRandomKernelJob(
              generation->at(i), spec_list)));
    }
    job_queue.UnlockQueue();
  } else {
    generation = vcsmc::ParseGenerationFile(FLAGS_seed_generation_file);
    if (!generation) {
      fprintf(stderr, "error parsing generation seed file.\n");
      return -1;
    }
    generation_size = generation->size();
    if (audio_spec_list) {
      job_queue.LockQueue();
      for (int i = 0; i < generation_size; ++i) {
        job_queue.EnqueueLocked(std::unique_ptr<vcsmc::Job>(
            new vcsmc::Kernel::ClobberSpecJob(generation->at(i),
                                              audio_spec_list)));
      }
      job_queue.UnlockQueue();
    }
  }

  // Initialize simulator global state.
  init_z26_global_tables();

  if (FLAGS_gperf_output_file != "") {
    printf("profiling enabled, saving output to %s.\n",
        FLAGS_gperf_output_file.c_str());
    ProfilerStart(FLAGS_gperf_output_file.c_str());
  }

  double target_error = strtod(FLAGS_target_error.c_str(), nullptr);
  int generation_count = 0;
  int streak = 0;
  double last_generation_score = 0.0;
  bool reroll = false;
  int reroll_count = 0;

  uint64 scoring_time_us = 0;
  uint64 scoring_count = 0;

  uint64 tourney_time_us = 0;
  uint64 tourney_count = 0;

  uint64 mutate_time_us = 0;
  uint64 mutate_count = 0;

  std::shared_ptr<vcsmc::Kernel> global_minimum = generation->at(0);

  auto epoch_time = std::chrono::high_resolution_clock::now();

  while (global_minimum->score() > target_error || generation_count == 0) {
    auto start_of_scoring = std::chrono::high_resolution_clock::now();

    // Score all unscored kernels in the current generation.
    job_queue.LockQueue();
    for (int j = 0; j < generation_size; ++j) {
      if (!generation->at(j)->score_valid()) {
        job_queue.EnqueueLocked(std::unique_ptr<vcsmc::Job>(
              new vcsmc::Kernel::ScoreKernelJob(generation->at(j),
                                                target_colors.get())));
        ++scoring_count;
      }
    }
    job_queue.UnlockQueue();

    // Wait for simulation to finish.
    job_queue.Finish();
    auto end_of_scoring = std::chrono::high_resolution_clock::now();

    scoring_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
        end_of_scoring - start_of_scoring).count();

    auto start_of_tourney = std::chrono::high_resolution_clock::now();

    // Conduct tournament based on scores while looking for new global minimum.
    job_queue.LockQueue();
    for (int j = 0; j < generation_size; ++j) {
      if (generation->at(j)->score() < global_minimum->score()) {
        global_minimum = generation->at(j);
      }
      std::array<uint32, vcsmc::kSeedSizeWords> seed;
      for (size_t k = 0; k < vcsmc::kSeedSizeWords; ++k)
        seed[k] = seed_engine();
      std::seed_seq tourney_seed(seed.begin(), seed.end());
      job_queue.EnqueueLocked(std::unique_ptr<vcsmc::Job>(new CompeteKernelJob(
          generation, generation->at(j), tourney_seed, FLAGS_tournament_size)));
      ++tourney_count;
    }
    job_queue.UnlockQueue();

    // Wait for tournament to finish.
    job_queue.Finish();
    auto end_of_tourney = std::chrono::high_resolution_clock::now();

    tourney_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
        end_of_tourney - start_of_tourney).count();

    // Sort generation by victories.
    std::sort(generation->begin(), generation->end(),
        [](std::shared_ptr<vcsmc::Kernel> a, std::shared_ptr<vcsmc::Kernel> b) {
          if (a->victories() == b->victories())
            return a->score() < b->score();

          return b->victories() < a->victories();
        });

    if ((generation_count % FLAGS_save_count) == 0) {
      auto now = std::chrono::high_resolution_clock::now();
      if (FLAGS_print_stats) {
        printf("gen: %7d leader: %016llx score: %14.4f "
               "sim: %7llu tourney: %7llu mutate: %7llu "
               "epoch: %7llu elapsed: %7llu"
               "%s\n",
            generation_count,
            generation->at(0)->fingerprint(),
            global_minimum->score(),
            scoring_count ? scoring_count * 1000000 / scoring_time_us : 0,
            tourney_count ? tourney_count * 1000000 / tourney_time_us : 0,
            mutate_count ? mutate_count * 1000000 / mutate_time_us : 0,
            std::chrono::duration_cast<std::chrono::seconds>(
              now - epoch_time).count(),
            std::chrono::duration_cast<std::chrono::seconds>(
                now - program_start_time).count(),
            reroll ? " reroll" : "");
      }
      reroll = false;
      scoring_count = 0;
      scoring_time_us = 0;
      tourney_count = 0;
      tourney_time_us = 0;
      mutate_count = 0;
      mutate_time_us = 0;
      SaveState(generation, global_minimum);
      epoch_time = now;
    }

    if (fabs(last_generation_score - global_minimum->score()) < 0.00001) {
      ++streak;
    } else {
      last_generation_score = global_minimum->score();
      streak = 0;
    }

    auto start_of_mutate = std::chrono::high_resolution_clock::now();
    if (FLAGS_stagnant_generation_count == 0 ||
        streak < FLAGS_stagnant_generation_count) {
      // Replace lowest-scoring half of generation with mutated versions of
      // highest-scoring half of generation.
      job_queue.LockQueue();
      for (int j = generation_size / 2; j < generation_size; ++j) {
        std::array<uint32, vcsmc::kSeedSizeWords> seed;
        for (size_t k = 0; k < vcsmc::kSeedSizeWords; ++k)
          seed[k] = seed_engine();
        std::seed_seq kernel_seed(seed.begin(), seed.end());
        generation->at(j).reset(new vcsmc::Kernel(kernel_seed));
        job_queue.EnqueueLocked(std::unique_ptr<vcsmc::Job>(
              new vcsmc::Kernel::MutateKernelJob(
                generation->at(j - (generation_size / 2)),
                generation->at(j),
                1)));
        ++mutate_count;
      }
      job_queue.UnlockQueue();
      job_queue.Finish();
    } else {
      ++reroll_count;
      if (reroll_count > FLAGS_stagnant_count_limit)
        break;
      reroll = true;
      streak = 0;
      vcsmc::Generation mutated_generation(
          new std::vector<std::shared_ptr<vcsmc::Kernel>>);
      job_queue.LockQueue();
      for (int j = 0; j < generation_size; ++j) {
        std::array<uint32, vcsmc::kSeedSizeWords> seed;
        for (size_t k = 0; k < vcsmc::kSeedSizeWords; ++k)
          seed[k] = seed_engine();
        std::seed_seq kernel_seed(seed.begin(), seed.end());
        mutated_generation->emplace_back(new vcsmc::Kernel(kernel_seed));
        job_queue.EnqueueLocked(std::unique_ptr<vcsmc::Job>(
              new vcsmc::Kernel::MutateKernelJob(
                generation->at(j),
                mutated_generation->at(j),
                FLAGS_stagnant_mutation_count)));
        ++mutate_count;
      }
      job_queue.UnlockQueue();
      job_queue.Finish();
      generation = mutated_generation;
    }
    auto end_of_mutate = std::chrono::high_resolution_clock::now();
    mutate_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
        end_of_mutate - start_of_mutate).count();

    ++generation_count;
    if (FLAGS_max_generation_number > 0 &&
        generation_count > FLAGS_max_generation_number) {
      break;
    }
  }

  if (FLAGS_gperf_output_file != "") {
    ProfilerStop();
  }

  SaveState(generation, global_minimum);

  if (FLAGS_append_kernel_binary != "") {
    vcsmc::AppendKernelBinary(global_minimum, FLAGS_append_kernel_binary);
  }

  return 0;
}
