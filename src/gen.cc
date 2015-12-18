// gen - uses Evolutionary Programming to evolve a series of Kernels that can
// generate some facsimile of the supplied input image.

#include <array>
#include <gflags/gflags.h>
#include <stdlib.h>
#include <random>

#include "image.h"
#include "image_file.h"
#include "color.h"
#include "job_queue.h"
#include "kernel.h"
#include "spec.h"

extern "C" {
#include <libz26/libz26.h>
}

DEFINE_int32(generation_size, 50000,
    "Number of individuals to keep in an evolutionary programing generation.");
DEFINE_int32(max_generation_number, 10000,
    "Number of generations of evolutionary programming to run.");
DEFINE_int32(tournament_size, 500,
    "Number of kernels each should compete against.");
DEFINE_int32(worker_threads, 0,
    "Number of threads to create to work in parallel, set to 0 to pick based "
    "on hardware.");

DEFINE_string(log_base_dir, "out/log", "Base directory for log entries.");
DEFINE_string(random_seed, "",
    "Hex string of seed for pseudorandom seed generation. If not provided one "
    "will be generated from hardware random device.");
DEFINE_string(spec_list_file, "asm/frame_spec.yaml",
    "Path to spec list yaml file.");
DEFINE_string(target_image_file, "",
    "Path to target image file to score kernels against.");

typedef std::shared_ptr<std::vector<std::shared_ptr<vcsmc::Kernel>>> Generation;

class CompeteKernelJob : public vcsmc::Job {
 public:
  CompeteKernelJob(
      const Generation generation,
      std::shared_ptr<vcsmc::Kernel> kernel,
      std::seed_seq& seed,
      size_t tourney_size)
      : generation_(generation),
        kernel_(kernel),
        engine_(seed),
        tourney_size_(tourney_size) {}
  void Execute() override {
    kernel_->ResetVictories();
    std::uniform_int_distribution<size_t> tourney_distro(0,
        generation_->size() - 1);
    for (size_t i = 0; i < tourney_size_; ++i) {
      size_t contestant_index = tourney_distro(engine_);
      if (generation_->at(contestant_index)->score() < kernel_->score())
        kernel_->AddVictory();
    }
  }
 private:
  const Generation generation_;
  std::shared_ptr<vcsmc::Kernel> kernel_;
  std::default_random_engine engine_;
  size_t tourney_size_;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  printf("parsing spec list %s.\n", FLAGS_spec_list_file.c_str());

  vcsmc::SpecList spec_list = vcsmc::ParseSpecListFile(FLAGS_spec_list_file);
  if (!spec_list) {
    fprintf(stderr, "Error parsing spec list file %s.\n",
        FLAGS_spec_list_file.c_str());
    return -1;
  }

  printf("generating initial population of %d individuals.\n",
      FLAGS_generation_size);

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
  Generation generation(new std::vector<std::shared_ptr<vcsmc::Kernel>>(
        FLAGS_generation_size));

  // Generate FLAGS_generation_size number of random individuals.
  for (int i = 0; i < FLAGS_generation_size; ++i) {
    std::array<uint32, vcsmc::kSeedSizeWords> seed;
    for (size_t j = 0; j < vcsmc::kSeedSizeWords; ++j)
      seed[j] = seed_engine();
    std::seed_seq kernel_seed(seed.begin(), seed.end());
    generation->emplace_back(new vcsmc::Kernel(kernel_seed));
    job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
        new vcsmc::Kernel::GenerateRandomKernelJob(
            generation->at(i), spec_list)));
  }

  printf("loading target image file %s.\n", FLAGS_target_image_file.c_str());

  std::unique_ptr<vcsmc::Image> target_image = vcsmc::ImageFile::Load(
      FLAGS_target_image_file);
  if (!target_image) {
    fprintf(stderr, "Error opening target_image_file \"%s\".",
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
  std::unique_ptr<double[]> target_lab(
      new double[vcsmc::kTargetFrameWidthPixels *
                 vcsmc::kFrameHeightPixels * 4]);
  for (size_t i = 0;
       i < (vcsmc::kTargetFrameWidthPixels * vcsmc::kFrameHeightPixels); ++i) {
    vcsmc::RGBAToLabA(reinterpret_cast<uint8*>(target_image->pixels() + i),
        target_lab.get() + (i * 4));
  }

  // Initialize simulator global state.
  init_z26_global_tables();

  for (int i = 0; i < FLAGS_max_generation_number; ++i) {

    printf("starting generation %d of %d.\n", i, FLAGS_max_generation_number);

    printf("    scoring kernels.\n");

    // Score all unscored kernels in the current generation.
    for (int j = 0; j < FLAGS_generation_size; ++j) {
      if (!generation->at(j)->score_valid()) {
        job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
              new vcsmc::Kernel::ScoreKernelJob(generation->at(j),
                                                target_lab.get())));
      }
    }

    // Wait for simulation to finish.
    job_queue.Finish();

    printf("  conducting tournament.\n");

    // Conduct tournament based on scores.
    for (int j = 0; j < FLAGS_generation_size; ++j) {
      std::array<uint32, vcsmc::kSeedSizeWords> seed;
      for (size_t k = 0; k < vcsmc::kSeedSizeWords; ++k)
        seed[j] = seed_engine();
      std::seed_seq tourney_seed(seed.begin(), seed.end());
      job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(new CompeteKernelJob(
          generation, generation->at(j), tourney_seed, FLAGS_tournament_size)));
    }
  }

  return 0;
}
