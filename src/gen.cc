// gen - uses Evolutionary Programming to evolve a series of Kernels that can
// generate some facsimile of the supplied input image.

#include <algorithm>
#include <array>
#include <cassert>
#include <gflags/gflags.h>
#include <gperftools/profiler.h>
#include <stdlib.h>
#include <random>

#include "bit_map.h"
#include "color.h"
#include "color_table.h"
#include "image.h"
#include "image_file.h"
#include "job_queue.h"
#include "kernel.h"
#include "serialization.h"
#include "spec.h"

extern "C" {
#include <libz26/libz26.h>
}

DEFINE_int32(generation_size, 1000,
    "Number of individuals to keep in an evolutionary programing generation.");
DEFINE_int32(max_generation_number, 0,
    "Maximum number of generations of evolutionary programming to run. If zero "
    "the program will run until the minimum error percentage target is met.");
DEFINE_int32(tournament_size, 100,
    "Number of kernels each should compete against.");
DEFINE_int32(worker_threads, 0,
    "Number of threads to create to work in parallel, set to 0 to pick based "
    "on hardware.");
DEFINE_int32(target_percent_error, 10,
    "Percentage above the minimum total error the program will attempt to "
    "reach.");
DEFINE_int32(save_count, 1000,
    "Number of generations to run before saving the results.");
DEFINE_int32(error_bitmap_multiplier, 5,
    "Multiplier to add to optional error weighting bitmap.");
DEFINE_int32(stagnant_generation_count, 250,
    "Number of generations without score change before randomizing entire "
    "cohort. Set to zero to disable.");
DEFINE_int32(stagnant_mutation_count, 16,
    "Number of mutations to apply to each cohort member when re-randomizing "
    "stagnant cohort.");

DEFINE_string(random_seed, "",
    "Hex string of seed for pseudorandom seed generation. If not provided one "
    "will be generated from hardware random device.");
DEFINE_string(spec_list_file, "asm/frame_spec.yaml",
    "Required - path to spec list yaml file.");
DEFINE_string(target_image_file, "",
    "Required - path to target image file to score kernels against.");
DEFINE_string(gperf_output_file, "",
    "Defining enables gperftools output to the provided profile path.");
DEFINE_string(seed_generation_file, "",
    "Optional file with saved generation data, to seed the first generation of "
    "the evolutionary programming algorithm. The generation size will be set "
    "to the number of kernels in the file. Any provided spec list will also be "
    "ignored as the kernels provide their own spec.");
DEFINE_string(generation_output_file, "out/generation.yaml",
    "Optional path to output yaml file for generation saves.");
DEFINE_string(image_output_file, "",
    "Optional file to save minimum simulated image to.");
DEFINE_string(input_error_weighting_bitmap, "",
    "Optional file to bitmap image of additional weight per-pixel to add to "
    "error distance computations.");
DEFINE_string(global_minimum_output_file, "out/minimum.yaml",
    "Required file path to save global minimum error kernel to.");
DEFINE_string(ideal_image_output_file, "out/ideal.png",
    "Optional file path to save ideal color fit image.");
DEFINE_string(audio_spec_list_file, "",
    "Optional file path for audio spec, will clobber any existing specs at "
    "same time in existing kernels.");

class ComputeColorErrorTableJob : public vcsmc::Job {
 public:
  ComputeColorErrorTableJob(
      const double* target_lab,
      std::vector<double>& error_table,
      size_t color_index,
      const vcsmc::BitMap* error_weights)
      : target_lab_(target_lab),
        error_table_(error_table),
        color_index_(color_index),
        error_weights_(error_weights) {}
  void Execute() override {
    const double* atari_lab = vcsmc::kAtariNTSCLabColorTable +
        (color_index_ * 4);
    for (size_t i = 0;
         i < vcsmc::kTargetFrameWidthPixels * vcsmc::kFrameHeightPixels; ++i) {
      double error = vcsmc::Ciede2k(target_lab_, atari_lab);
      if (error_weights_ &&
          error_weights_->bit(i % vcsmc::kTargetFrameWidthPixels,
                              i / vcsmc::kTargetFrameWidthPixels)) {
        error *= static_cast<double>(FLAGS_error_bitmap_multiplier);
      }
      target_lab_ += 4;
      error_table_.push_back(error);
    }
  }
 private:
  const double* target_lab_;
  std::vector<double>& error_table_;
  size_t color_index_;
  const vcsmc::BitMap* error_weights_;
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
    generation = vcsmc::ParseGenerationFile(FLAGS_seed_generation_file);
    if (!generation) {
      fprintf(stderr, "error parsing generation seed file.\n");
      return -1;
    }
    generation_size = generation->size();
    if (audio_spec_list) {
      for (int i = 0; i < generation_size; ++i) {
        job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
            new vcsmc::Kernel::ClobberSpecJob(generation->at(i),
                                              audio_spec_list)));
      }
    }
  }

  std::unique_ptr<vcsmc::Image> target_image = vcsmc::LoadImage(
      FLAGS_target_image_file);
  if (!target_image) {
    fprintf(stderr, "error opening target_image_file \"%s\".\n",
        FLAGS_target_image_file.c_str());
    return -1;
  }
  if (target_image->width() != vcsmc::kTargetFrameWidthPixels ||
      target_image->height() != vcsmc::kFrameHeightPixels) {
    fprintf(stderr, "Bad image dimensions %dx%d not %dx%d in %s\n",
        target_image->width(),
        target_image->height(),
        vcsmc::kTargetFrameWidthPixels,
        vcsmc::kFrameHeightPixels,
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

  std::unique_ptr<vcsmc::BitMap> error_weights;
  if (FLAGS_input_error_weighting_bitmap != "") {
    error_weights = vcsmc::BitMap::Load(FLAGS_input_error_weighting_bitmap);
    if (!error_weights) {
      fprintf(stderr, "error opening error weight image file %s.\n",
          FLAGS_input_error_weighting_bitmap.c_str());
      return -1;
    }
    if (error_weights->width() != vcsmc::kTargetFrameWidthPixels ||
        error_weights->height() != vcsmc::kFrameHeightPixels) {
      fprintf(stderr,
          "Bad error weight image dimensions %dx%d not %dx%d in %s\n",
          error_weights->width(),
          error_weights->height(),
          vcsmc::kTargetFrameWidthPixels,
          vcsmc::kFrameHeightPixels,
          FLAGS_input_error_weighting_bitmap.c_str());
      return -1;
    }
  }

  vcsmc::Kernel::ScoreKernelJob::ColorDistances color_distances(128);
  for (size_t i = 0; i < vcsmc::kNTSCColors; ++i) {
    job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(new ComputeColorErrorTableJob(
        target_lab.get(), color_distances[i], i, error_weights.get())));
  }

  job_queue.Finish();

  // Compute theoretical minimum error.
  double min_total_error = 0.0;
  std::unique_ptr<vcsmc::Image> ideal_image;
  if (FLAGS_ideal_image_output_file != "") {
    ideal_image.reset(new vcsmc::Image(vcsmc::kTargetFrameWidthPixels,
          vcsmc::kFrameHeightPixels));
  }
  for (size_t i = 0;
      i < vcsmc::kTargetFrameWidthPixels * vcsmc::kFrameHeightPixels; i += 2) {
    double min_pixel_error = color_distances[0][i] + color_distances[0][i + 1];
    size_t min_pixel_color = 0;
    for (size_t j = 1; j < vcsmc::kNTSCColors; ++j) {
      double sum = color_distances[j][i] + color_distances[j][i + 1];
      if (sum < min_pixel_error) {
        min_pixel_error = sum;
        min_pixel_color = j;
      }
    }
    min_total_error += min_pixel_error;
    if (ideal_image) {
      ideal_image->pixels_writeable()[i] =
          vcsmc::kAtariNTSCABGRColorTable[min_pixel_color];
      ideal_image->pixels_writeable()[i + 1] =
          vcsmc::kAtariNTSCABGRColorTable[min_pixel_color];
    }
  }
  if (ideal_image) {
    if (!vcsmc::SaveImage(ideal_image.get(),
          FLAGS_ideal_image_output_file)) {
      fprintf(stderr, "Error saving ideal image output file.\n");
      return -1;
    }
  }

  // Initialize simulator global state.
  init_z26_global_tables();

  if (FLAGS_gperf_output_file != "") {
    printf("profiling enabled, saving output to %s.\n",
        FLAGS_gperf_output_file.c_str());
    ProfilerStart(FLAGS_gperf_output_file.c_str());
  }

  double percent_error = 0.0;
  double target_percent_error = static_cast<double>(FLAGS_target_percent_error);
  int generation_count = 0;
  int streak = 0;
  double last_generation_score = 0.0;
  bool reroll = false;

  std::shared_ptr<vcsmc::Kernel> global_minimum = generation->at(0);

  while (percent_error > target_percent_error || generation_count == 0) {
    // Score all unscored kernels in the current generation.
    for (int j = 0; j < generation_size; ++j) {
      if (!generation->at(j)->score_valid()) {
        job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
              new vcsmc::Kernel::ScoreKernelJob(generation->at(j),
                                                color_distances)));
      }
    }

    // Wait for simulation to finish.
    job_queue.Finish();

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

    // Sort generation by victories.
    std::sort(generation->begin(), generation->end(),
        [](std::shared_ptr<vcsmc::Kernel> a, std::shared_ptr<vcsmc::Kernel> b) {
          if (a->victories() == b->victories())
            return a->score() < b->score();

          return b->victories() < a->victories();
        });

    if (generation->at(0)->score() < global_minimum->score()) {
      global_minimum = generation->at(0);
    }

    percent_error = ((generation->at(0)->score() - min_total_error) /
        min_total_error) * 100.0;

    if ((generation_count % FLAGS_save_count) == 0) {
      // Report statistics and save generation to disk.
      // printf("gen: %7d leader: %016llx score: %6.2f%% %s\n",
      //    generation_count,
      //    generation->at(0)->fingerprint(),
      //    percent_error,
      //    reroll ? "*" : "");
      reroll = false;
      SaveState(generation, global_minimum);
    }

    if (fabs(last_generation_score - percent_error) < 0.00001) {
      ++streak;
    } else {
      last_generation_score = percent_error;
      streak = 0;
    }

    if (FLAGS_stagnant_generation_count == 0 ||
        streak < FLAGS_stagnant_generation_count) {
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
                1)));
      }
      job_queue.Finish();
    } else {
      reroll = true;
      streak = 0;
      vcsmc::Generation mutated_generation(
          new std::vector<std::shared_ptr<vcsmc::Kernel>>);
      for (int j = 0; j < generation_size; ++j) {
        std::array<uint32, vcsmc::kSeedSizeWords> seed;
        for (size_t k = 0; k < vcsmc::kSeedSizeWords; ++k)
          seed[k] = seed_engine();
        std::seed_seq kernel_seed(seed.begin(), seed.end());
        mutated_generation->emplace_back(new vcsmc::Kernel(kernel_seed));
        job_queue.Enqueue(std::unique_ptr<vcsmc::Job>(
              new vcsmc::Kernel::MutateKernelJob(
                generation->at(j),
                mutated_generation->at(j),
                FLAGS_stagnant_mutation_count)));
      }
      job_queue.Finish();
      generation = mutated_generation;
    }

    ++generation_count;
    if (FLAGS_max_generation_number > 0 &&
        generation_count > FLAGS_max_generation_number) {
      // printf("max generation count limit %d hit, exiting.\n",
      //    FLAGS_max_generation_number);
      break;
    }
  }

  if (FLAGS_gperf_output_file != "") {
    ProfilerStop();
  }

  SaveState(generation, global_minimum);

  return 0;
}
