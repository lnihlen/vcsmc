// gen - uses Evolutionary Programming to evolve a series of Kernels that can
// generate some facsimile of the supplied input image.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <gflags/gflags.h>
#include <gperftools/profiler.h>
#include <stdlib.h>
#include <random>

#include "image.h"
#include "image_file.h"
#include "color.h"
#include "color_table.h"
#include "job_queue.h"
#include "kernel.h"
#include "serialization.h"
#include "spec.h"

extern "C" {
#include <libz26/libz26.h>
}

DEFINE_int32(generation_size, 1000,
    "Number of individuals to keep in an evolutionary programing generation.");
DEFINE_int32(max_generation_number, 100000,
    "Number of generations of evolutionary programming to run.");
DEFINE_int32(tournament_size, 100,
    "Number of kernels each should compete against.");
DEFINE_int32(worker_threads, 0,
    "Number of threads to create to work in parallel, set to 0 to pick based "
    "on hardware.");
// TODO: invert logic here, something like max - (i * stale_mutations)
DEFINE_int32(max_mutations, 5,
    "Maximum number of mutations applied per-kernel to a single generation, "
    "based on number of generations same kernel has won.");
DEFINE_int32(stale_mutations_multiplier, 1,
    "Number of mutations to add per generation of stale evolution.");

DEFINE_string(log_base_dir, "", "Base directory for log entries.");
DEFINE_string(random_seed, "",
    "Hex string of seed for pseudorandom seed generation. If not provided one "
    "will be generated from hardware random device.");
DEFINE_string(spec_list_file, "asm/frame_spec.yaml",
    "Path to spec list yaml file.");
DEFINE_string(target_image_file, "",
    "Required path to target image file to score kernels against.");
DEFINE_string(gperf_output_file, "",
    "Defining enables gperftools output to the provided profile path.");
DEFINE_string(seed_generation_file, "",
    "Optional file with saved generation data, to seed the first generation of "
    "the evolutionary programming algorithm. The generation size will be set "
    "to the number of kernels in the file. Any provided spec list will also be "
    "ignored as the kernels provide their own spec.");
DEFINE_string(final_generation_file, "out/generation.yaml",
    "Required path to output yaml file for final generation.");

std::string FormatDuration(const std::chrono::duration<uint64,
    std::milli>& duration) {
  auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
  auto minutes = std::chrono::duration_cast<std::chrono::minutes>(
      duration % std::chrono::hours(1));
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
      duration % std::chrono::minutes(1));
  auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(
      duration % std::chrono::seconds(1));
  char dur_buf[64];
  snprintf(dur_buf, 64, "%ld:%02ld:%02lld.%03lld", hours.count(),
      minutes.count(), seconds.count(), msec.count());
  return std::string(dur_buf);
}

class ComputeColorErrorTableJob : public vcsmc::Job {
 public:
  ComputeColorErrorTableJob(
      const double* target_lab,
      std::vector<double>& error_table,
      size_t color_index)
      : target_lab_(target_lab),
        error_table_(error_table),
        color_index_(color_index),
        call_count_(0) {}
  void Execute() override {
    ++call_count_;
    assert(call_count_ == 1);
    const double* atari_lab = vcsmc::kAtariNTSCLabColorTable +
        (color_index_ * 4);
    for (size_t i = 0;
         i < vcsmc::kTargetFrameWidthPixels * vcsmc::kFrameHeightPixels; ++i) {
      double error = vcsmc::Ciede2k(target_lab_, atari_lab);
      target_lab_ += 4;
      error_table_.push_back(error);
    }
  }
 private:
  const double* target_lab_;
  std::vector<double>& error_table_;
  size_t color_index_;
  size_t call_count_;
};

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

int main(int argc, char* argv[]) {
  std::chrono::time_point<std::chrono::system_clock> program_start =
      std::chrono::system_clock::now();
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

  std::seed_seq master_seed(seed_array.begin(), seed_array.end());
  std::default_random_engine seed_engine(master_seed);

  vcsmc::JobQueue job_queue(FLAGS_worker_threads);
  vcsmc::Generation generation;

  int generation_size = FLAGS_generation_size;
  if (FLAGS_seed_generation_file == "") {
    printf("parsing spec list %s.\n", FLAGS_spec_list_file.c_str());

    vcsmc::SpecList spec_list = vcsmc::ParseSpecListFile(FLAGS_spec_list_file);
    if (!spec_list) {
      fprintf(stderr, "Error parsing spec list file %s.\n",
          FLAGS_spec_list_file.c_str());
      return -1;
    }

    printf("generating initial population of %d individuals.\n",
        generation_size);
    generation.reset(new std::vector<std::shared_ptr<vcsmc::Kernel>>);
    // Generate FLAGS_generation_size number of random individuals.
    for (int i = 0; i < generation_size; ++i) {
      std::array<uint32, vcsmc::kSeedSizeWords> seed;
      for (size_t j = 0; j < vcsmc::kSeedSizeWords; ++j)
        seed[j] = seed_engine();
      std::seed_seq kernel_seed(seed.begin(), seed.end());
      generation->emplace_back(new vcsmc::Kernel(kernel_seed));
      job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
          new vcsmc::Kernel::GenerateRandomKernelJob(
              generation->at(i), spec_list)));
    }
  } else {
    printf("loading initial population from %s.\n",
        FLAGS_seed_generation_file.c_str());
    generation = vcsmc::ParseGenerationFile(FLAGS_seed_generation_file);
    if (!generation) {
      fprintf(stderr, "error parsing generation seed file.\n");
      return -1;
    }
    generation_size = generation->size();
    printf("  loaded %d kernels.\n", generation_size);
  }

  printf("loading target image file %s.\n", FLAGS_target_image_file.c_str());

  std::unique_ptr<vcsmc::Image> target_image = vcsmc::ImageFile::Load(
      FLAGS_target_image_file);
  if (!target_image) {
    fprintf(stderr, "error opening target_image_file \"%s\".",
        FLAGS_target_image_file.c_str());
    return -1;
  }
  if (target_image->width() != vcsmc::kTargetFrameWidthPixels ||
      target_image->height() != vcsmc::kFrameHeightPixels) {
    fprintf(stderr, "Bad image dimensions %dx%d not %dx%d in %s",
        target_image->width(), target_image->height(),
        vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels,
        FLAGS_target_image_file.c_str());
    return -1;
  }

  // Convert target image to Lab colors for scoring.
  std::shared_ptr<double> target_lab(
      new double[vcsmc::kTargetFrameWidthPixels *
                 vcsmc::kFrameHeightPixels * 4]);
  for (size_t i = 0;
       i < (vcsmc::kTargetFrameWidthPixels * vcsmc::kFrameHeightPixels); ++i) {
    vcsmc::RGBAToLabA(reinterpret_cast<uint8*>(target_image->pixels() + i),
        target_lab.get() + (i * 4));
  }

  printf("building error tables for target image.\n");

  vcsmc::Kernel::ScoreKernelJob::ColorDistances color_distances(128);
  for (size_t i = 0; i < vcsmc::kNTSCColors; ++i) {
    job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(new ComputeColorErrorTableJob(
        target_lab.get(), color_distances[i], i)));
  }

  job_queue.Finish();

  // Compute theoretical minimum error.
  double min_total_error = 0.0;
  for (size_t i = 0;
      i < vcsmc::kTargetFrameWidthPixels * vcsmc::kFrameHeightPixels; i += 2) {
    double min_pixel_error = color_distances[0][i] + color_distances[0][i + 1];
    for (size_t j = 1; j < vcsmc::kNTSCColors; ++j) {
      double sum = color_distances[j][i] + color_distances[j][i + 1];
      min_pixel_error = std::min(min_pixel_error, sum);
    }
    min_total_error += min_pixel_error;
  }

  printf("minimum total theoretical error: %f\n", min_total_error);

  // Initialize simulator global state.
  init_z26_global_tables();

  if (FLAGS_gperf_output_file != "") {
    printf("profiling enabled, saving output to %s.\n",
        FLAGS_gperf_output_file.c_str());
    ProfilerStart(FLAGS_gperf_output_file.c_str());
  }

  int streak = 0;
  uint64 last_champion = 0;

  for (int i = 0; i < FLAGS_max_generation_number; ++i) {
    std::chrono::time_point<std::chrono::system_clock> loop_start =
        std::chrono::system_clock::now();

    printf("starting generation %d of %d.\n", i, FLAGS_max_generation_number);

    printf("    scoring kernels.\n");

    std::chrono::time_point<std::chrono::system_clock> sim_start =
        std::chrono::system_clock::now();

    uint32 score_count = 0;
    // Score all unscored kernels in the current generation.
    for (int j = 0; j < generation_size; ++j) {
      if (!generation->at(j)->score_valid()) {
        job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
              new vcsmc::Kernel::ScoreKernelJob(generation->at(j),
                                                color_distances)));
        ++score_count;
      }
    }

    // Wait for simulation to finish.
    job_queue.Finish();

    std::chrono::time_point<std::chrono::system_clock> sim_finish =
        std::chrono::system_clock::now();
    std::chrono::duration<uint64, std::milli> sim_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            sim_finish - sim_start);
    std::string dur_str = FormatDuration(sim_duration);
    uint64 fps = (static_cast<uint64>(score_count) * 1000) /
        sim_duration.count();
    printf("    simulation took %s, fps: %llu.\n", dur_str.c_str(), fps);

    printf("    conducting tournament.\n");

    // Conduct tournament based on scores.
    for (int j = 0; j < generation_size; ++j) {
      std::array<uint32, vcsmc::kSeedSizeWords> seed;
      for (size_t k = 0; k < vcsmc::kSeedSizeWords; ++k)
        seed[k] = seed_engine();
      std::seed_seq tourney_seed(seed.begin(), seed.end());
      job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(new CompeteKernelJob(
          generation, generation->at(j), tourney_seed, FLAGS_tournament_size)));
    }

    // Wait for tournament to finish.
    job_queue.Finish();

    printf("    sorting results.\n");

    // Sort generation by victories.
    std::sort(generation->begin(), generation->end(),
        [](std::shared_ptr<vcsmc::Kernel> a, std::shared_ptr<vcsmc::Kernel> b) {
          return b->victories() < a->victories();
        });

    if (generation->at(0)->fingerprint() == last_champion) {
      ++streak;
    } else {
      last_champion = generation->at(0)->fingerprint();
      streak = 0;
    }

    double percent_error = ((generation->at(0)->score() - min_total_error) /
        min_total_error) * 100.0;

    // Report statistics and save champion image to disk.
    printf("    grand champion: %016llx, streak: %d, with score: %f, "
           "%.02f percent of theoretical minimum.\n",
        generation->at(0)->fingerprint(), streak, generation->at(0)->score(),
        percent_error);

    int mutations = std::min(FLAGS_max_mutations,
        1 + (streak * FLAGS_stale_mutations_multiplier));
    printf("    mutating generation %d times each.\n", mutations);
    // Replace lowest-scoring half of generation with mutated versions of
    // highest-scoring half of generation.
    for (int j = generation_size / 2; j < generation_size; ++j) {
      std::array<uint32, vcsmc::kSeedSizeWords> seed;
      for (size_t k = 0; k < vcsmc::kSeedSizeWords; ++k)
        seed[k] = seed_engine();
      std::seed_seq kernel_seed(seed.begin(), seed.end());
      generation->at(j).reset(new vcsmc::Kernel(kernel_seed));
      job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
            new vcsmc::Kernel::MutateKernelJob(
              generation->at(j - (generation_size / 2)),
              generation->at(j),
              mutations)));
    }

    job_queue.Finish();

    std::chrono::time_point<std::chrono::system_clock> loop_finish =
        std::chrono::system_clock::now();
    std::chrono::duration<uint64, std::milli> loop_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            loop_finish - loop_start);
    dur_str = FormatDuration(loop_duration);
    std::chrono::duration<uint64, std::milli> eta = loop_duration *
        (FLAGS_max_generation_number - i - 1);
    std::string eta_str = FormatDuration(eta);
    printf("    loop took %s, eta: %s.\n", dur_str.c_str(), eta_str.c_str());
  }

  if (FLAGS_gperf_output_file != "") {
    ProfilerStop();
  }

  printf("saving final generation to %s.\n",
      FLAGS_final_generation_file.c_str());

  if (!vcsmc::SaveGenerationFile(generation, FLAGS_final_generation_file)) {
    fprintf(stderr, "error saving final generation file %s\n",
        FLAGS_final_generation_file.c_str());
  }

  char kernel_image_path_buf[128];
  snprintf(kernel_image_path_buf, 128,
      "%s/%05d-%016llx.png",
      FLAGS_log_base_dir.c_str(),
      FLAGS_max_generation_number,
      generation->at(0)->fingerprint());
  generation->at(0)->SaveImage(std::string(kernel_image_path_buf));

  std::chrono::time_point<std::chrono::system_clock> program_finish =
      std::chrono::system_clock::now();
  std::chrono::duration<uint64, std::milli> total_program_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          program_finish - program_start);
  std::string dur_str = FormatDuration(total_program_duration);
  printf("total program duration %s.\n", dur_str.c_str());

  return 0;
}
