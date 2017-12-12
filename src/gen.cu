// gen - uses Evolutionary Programming to evolve a series of Kernels that can
// generate some facsimile of the supplied input image.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <gperftools/profiler.h>
#include <random>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "tbb/tbb.h"

#include "atari_ntsc_nyuv_color_table.h"
#include "bit_map.h"
#include "color.h"
#include "cuda_utils.h"
#include "image.h"
#include "image_file.h"
#include "kernel.h"
#include "mssim.h"
#include "serialization.h"
#include "spec.h"
#include "tls_prng.h"

extern "C" {
#include "libz26/libz26.h"
}

const size_t kSimSkipLines = 23;
const size_t kBlockSumBufferSize = 120;

DEFINE_bool(print_stats, true,
    "If true gen will print generation statistics to stdio after every "
    "save_count generations.");

DEFINE_int32(generation_size, 1000,
    "Number of individuals to keep in an evolutionary programming generation.");
DEFINE_int32(max_generation_number, 0,
    "Maximum number of generations of evolutionary programming to run. If zero "
    "the program will run until the minimum error percentage target is met.");
DEFINE_int32(tournament_size, 100,
    "Number of kernels each should compete against.");
DEFINE_int32(save_count, 1000,
    "Number of generations to run before saving the results.");
DEFINE_int32(stagnant_generation_count, 250,
    "Number of generations without score change before randomizing entire "
    "cohort. Set to zero to disable.");
DEFINE_int32(stagnant_mutation_count, 16,
    "Number of mutations to apply to each cohort member when re-randomizing "
    "stagnant cohort.");
DEFINE_int32(stagnant_count_limit, 0,
    "If nonzero, terminate after the supplied number of re-randomizations.");

DEFINE_string(input_image_file, "",
    "Required - the input png file to fit against.");
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
DEFINE_string(target_error, "0.0",
    "Error at which to stop optimizing.");
DEFINE_string(append_kernel_binary, "", "Path to append kernel binary.");

struct MssimScoreState {
 public:
  MssimScoreState() {
    cudaError_t result;
    cudaStream_t stream;
    result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(result == cudaSuccess);

    result = cudaMalloc(&sim_nyuv_device, vcsmc::kNyuvBufferSizeBytes);
    assert(result == cudaSuccess);
    result = cudaMalloc(&sim_mean_device, vcsmc::kNyuvBufferSizeBytes);
    assert(result == cudaSuccess);
    result = cudaMalloc(&sim_stddevsq_device, vcsmc::kNyuvBufferSizeBytes);
    assert(result == cudaSuccess);
    result = cudaMalloc(&sim_cov_device, vcsmc::kNyuvBufferSizeBytes);
    assert(result == cudaSuccess);
    result = cudaMalloc(&ssim_device, sizeof(float) *
        vcsmc::kTargetFrameWidthPixels * vcsmc::kFrameHeightPixels);
    assert(result == cudaSuccess);
    result = cudaMalloc(&block_sum_device, sizeof(float) * kBlockSumBufferSize);
    assert(result == cudaSuccess);
    result = cudaMalloc(&final_sum_device, sizeof(float));
    assert(result == cudaSuccess);
  }

  ~MssimScoreState() {
    // Clear any pending work on this thread.
    cudaStreamSynchronize(stream);

    cudaFree(sim_nyuv_device);
    cudaFree(sim_mean_device);
    cudaFree(sim_stddevsq_device);
    cudaFree(sim_cov_device);
    cudaFree(ssim_device);
    cudaFree(block_sum_device);

    cudaStreamDestroy(stream);
  }
  MssimScoreState(const MssimScoreState&) = delete;

  cudaStream_t stream;
  float4* sim_nyuv_device = nullptr;
  float4* sim_mean_device = nullptr;
  float4* sim_stddevsq_device = nullptr;
  float4* sim_cov_device = nullptr;
  float* ssim_device = nullptr;
  float* block_sum_device = nullptr;
  float* final_sum_device = nullptr;
};

struct KernelScore {
 public:
  KernelScore()
    : sim_frame(new uint8[kLibZ26ImageSizeBytes]),
      block_sums(new float[kBlockSumBufferSize]),
      sim_nyuv(new float[vcsmc::kNyuvBufferSize]) {}

  std::shared_ptr<uint8> sim_frame;
  std::shared_ptr<float> block_sums;
  std::shared_ptr<float> sim_nyuv;
  float score = 1.0;
  uint32 victories = 0;
  bool score_final = false;
};

struct KernelHash {
  size_t operator()(const std::shared_ptr<vcsmc::Kernel>& kernel) const {
    return static_cast<size_t>(kernel->fingerprint());
  }
};

struct KernelEqual {
  bool operator()(const std::shared_ptr<vcsmc::Kernel>& kernel_a,
                  const std::shared_ptr<vcsmc::Kernel>& kernel_b) const {
          return kernel_a->fingerprint() == kernel_b->fingerprint();
  }
};

typedef tbb::enumerable_thread_specific<MssimScoreState> MssimScoreStateList;
// Sweeping an interesting thing under the rug for now - this map uses 32-bit
// keys (size_t) but the fingerprints on the Kernel are 64-bit.
typedef tbb::concurrent_unordered_map<std::shared_ptr<vcsmc::Kernel>,
        KernelScore, KernelHash, KernelEqual> ScoreMap;

// Given a pointer to a completely empty Kernel this Job will populate it with
// totally random bytecode.
class GenerateRandomKernelJob {
 public:
  GenerateRandomKernelJob(vcsmc::Generation generation,
                          vcsmc::SpecList specs,
                          vcsmc::TlsPrngList& tls_prng_list)
      : generation_(generation),
        specs_(specs),
        tls_prng_list_(tls_prng_list) {}
  void operator()(const tbb::blocked_range<size_t>& r) const {
    vcsmc::TlsPrngList::reference tls_prng = tls_prng_list_.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      std::shared_ptr<vcsmc::Kernel> kernel = generation_->at(i);
      kernel->GenerateRandom(specs_, tls_prng);
    }
  }

 private:
  vcsmc::Generation generation_;
  const vcsmc::SpecList specs_;
  vcsmc::TlsPrngList& tls_prng_list_;
};

// Given an existing Kernel and a new list of specs (typically audio) this
// will clobber the existing specs and regenerate the bytecode.
class ClobberSpecJob {
 public:
  ClobberSpecJob(vcsmc::Generation generation,
                 vcsmc::SpecList specs)
      : generation_(generation),
        specs_(specs) {}
  void operator()(const tbb::blocked_range<size_t>& r) const {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      std::shared_ptr<vcsmc::Kernel> kernel = generation_->at(i);
      kernel->ClobberSpec(specs_);
    }
  }
 private:
  vcsmc::Generation generation_;
  const vcsmc::SpecList specs_;
};

// Given a provided reference kernel, generate the target kernel as a copy of
// the reference with the provided number of random mutations. Should iterate
// over the first half of |generation|, which will use that as source material
// and target the latter half of the array for copy and mutation.
class MutateKernelJob {
 public:
  MutateKernelJob(vcsmc::Generation source_generation,
                  vcsmc::Generation target_generation,
                  size_t target_index_offset,
                  size_t number_of_mutations,
                  ScoreMap& score_map,
                  vcsmc::TlsPrngList& tls_prng_list)
      : source_generation_(source_generation),
        target_generation_(target_generation),
        target_index_offset_(target_index_offset),
        number_of_mutations_(number_of_mutations),
        score_map_(score_map),
        tls_prng_list_(tls_prng_list) {}
  void operator()(const tbb::blocked_range<size_t>& r) const {
    vcsmc::TlsPrngList::reference tls_prng = tls_prng_list_.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      std::shared_ptr<vcsmc::Kernel> original = source_generation_->at(i);
      std::shared_ptr<vcsmc::Kernel> target = original->Clone();
      target_generation_->at(i + target_index_offset_) = target;

      // Now do the mutations to the target.
      for (size_t j = 0; j < number_of_mutations_; ++j)
        target->Mutate(tls_prng);

      target->RegenerateBytecode();

      // Sometimes the mutation currently results in a situation where the
      // fingerprint doesn't change between target and original, or the
      // fingerprint does but the hash table still collides because we are
      // disregarding the 32 most significant bits of the fingerprint in the
      // hash. This results in hash table collisions and other sadness. TODO:
      // figure out why this happens with better Kernel unit testing on
      // Kernel::Muate().
      while (target->fingerprint() == original->fingerprint() ||
             score_map_.find(target) != score_map_.end()) {
        for (size_t j = 0; j < number_of_mutations_; ++j)
          target->Mutate(tls_prng);

        target->RegenerateBytecode();
      }
    }
 }

 private:
  vcsmc::Generation source_generation_;
  vcsmc::Generation target_generation_;
  const size_t target_index_offset_;
  const size_t number_of_mutations_;
  ScoreMap& score_map_;
  vcsmc::TlsPrngList& tls_prng_list_;
};

class SimulateKernelJob {
 public:
  SimulateKernelJob(vcsmc::Generation generation,
                    const float4* target_nyuv_device,
                    const float4* target_mean_device,
                    const float4* target_stddevsq_device,
                    ScoreMap& score_map,
                    MssimScoreStateList& mssim_score_state_list)
    : generation_(generation),
      target_nyuv_device_(target_nyuv_device),
      target_mean_device_(target_mean_device),
      target_stddevsq_device_(target_stddevsq_device),
      score_map_(score_map),
      mssim_score_state_list_(mssim_score_state_list) {}
  void operator()(const tbb::blocked_range<size_t>& r) const {
    MssimScoreStateList::reference score_state =
          mssim_score_state_list_.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      std::shared_ptr<vcsmc::Kernel> kernel = generation_->at(i);
      KernelScore kernel_score;

      simulate_single_frame(kernel->bytecode(), kernel->bytecode_size(),
          kernel_score.sim_frame.get());

      // Convert simulated colors to nYUV for MSSIM computation.
      float* nyuv = kernel_score.sim_nyuv.get();
      // Zero out the image, to also zero out the padding.
      std::memset(nyuv, 0, vcsmc::kNyuvBufferSizeBytes);
      uint8* frame_pointer = kernel_score.sim_frame.get() +
          (kLibZ26ImageWidth * kSimSkipLines);
      for (size_t j = 0; j < vcsmc::kFrameHeightPixels; ++j) {
        for (size_t k = 0; k < vcsmc::kFrameWidthPixels; ++k) {
          uint8 col = *frame_pointer;
          if (col < 128) {
            uint32 nyuv_index = col * 3;
            nyuv[0] = vcsmc::kAtariNtscNyuvColorTable[nyuv_index];
            nyuv[1] = vcsmc::kAtariNtscNyuvColorTable[nyuv_index + 1];
            nyuv[2] = vcsmc::kAtariNtscNyuvColorTable[nyuv_index + 2];
            nyuv[3] = 1.0;
            nyuv[4] = vcsmc::kAtariNtscNyuvColorTable[nyuv_index];
            nyuv[5] = vcsmc::kAtariNtscNyuvColorTable[nyuv_index + 1];
            nyuv[6] = vcsmc::kAtariNtscNyuvColorTable[nyuv_index + 2];
            nyuv[7] = 1.0;
          } else {
            nyuv[0] = 0.0;
            nyuv[1] = 0.0;
            nyuv[2] = 0.0;
            nyuv[3] = 1.0;
            nyuv[4] = 0.0;
            nyuv[5] = 0.0;
            nyuv[6] = 0.0;
            nyuv[7] = 1.0;
          }
          frame_pointer += 2;
          nyuv += 8;
        }

        // Skip padding on right.
        nyuv += vcsmc::kWindowSize * 4;
      }

      dim3 image_dim_block(16, 16);
      dim3 padded_image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                                 (vcsmc::kFrameHeightPixels / 16) + 1);
      dim3 image_dim_grid((vcsmc::kTargetFrameWidthPixels / 16),
                          (vcsmc::kFrameHeightPixels / 16));
      dim3 sum_dim_block(256);
      dim3 sum_dim_grid(kBlockSumBufferSize);

      cudaMemcpyAsync(score_state.sim_nyuv_device,
                      kernel_score.sim_nyuv.get(),
                      vcsmc::kNyuvBufferSizeBytes,
                      cudaMemcpyHostToDevice,
                      score_state.stream);
      vcsmc::ComputeLocalMean<<<image_dim_block, padded_image_dim_grid, 0,
          score_state.stream>>>(score_state.sim_nyuv_device,
                                score_state.sim_mean_device);
      vcsmc::ComputeLocalStdDevSquared<<<image_dim_block, padded_image_dim_grid,
          0, score_state.stream>>>(score_state.sim_nyuv_device,
                                   score_state.sim_mean_device,
                                   score_state.sim_stddevsq_device);
      vcsmc::ComputeLocalCovariance<<<image_dim_block, padded_image_dim_grid, 0,
          score_state.stream>>>(score_state.sim_nyuv_device,
                                score_state.sim_mean_device,
                                target_nyuv_device_,
                                target_mean_device_,
                                score_state.sim_cov_device);
      vcsmc::ComputeSSIM<<<image_dim_block, image_dim_grid, 0,
          score_state.stream>>>(score_state.sim_mean_device,
                                score_state.sim_stddevsq_device,
                                target_mean_device_,
                                target_stddevsq_device_,
                                score_state.sim_cov_device,
                                score_state.ssim_device);
      vcsmc::ComputeBlockSum<<<sum_dim_block, sum_dim_grid,
          kBlockSumBufferSize * sizeof(float), score_state.stream>>>(
          score_state.ssim_device, score_state.block_sum_device);
      cudaMemcpyAsync(kernel_score.block_sums.get(),
                      score_state.block_sum_device,
                      sizeof(float) * kBlockSumBufferSize,
                      cudaMemcpyDeviceToHost,
                      score_state.stream);

      score_map_.insert(std::make_pair(kernel, kernel_score));
    }
  }

 private:
  vcsmc::Generation generation_;
  const float4* target_nyuv_device_;
  const float4* target_mean_device_;
  const float4* target_stddevsq_device_;
  ScoreMap& score_map_;
  MssimScoreStateList& mssim_score_state_list_;
};

// As we now use CUDA to compute almost all of the score asynchronously we
// wait for all GPU work to complete and then call this on each kernel to
// finish computation.
class FinalizeScoreJob {
 public:
  FinalizeScoreJob(vcsmc::Generation generation,
                   ScoreMap& score_map,
                   MssimScoreStateList& mssim_score_state_list)
      : generation_(generation),
        score_map_(score_map),
        mssim_score_state_list_(mssim_score_state_list) {}
  void operator()(const tbb::blocked_range<size_t>& r) const {
    MssimScoreStateList::reference score_state =
        mssim_score_state_list_.local();
    cudaError_t result = cudaStreamSynchronize(score_state.stream);
    assert(result == cudaSuccess);
    for (size_t i = r.begin(); i < r.end(); ++i) {
      std::shared_ptr<vcsmc::Kernel> kernel = generation_->at(i);
      ScoreMap::iterator kernel_score = score_map_.find(kernel);
      assert(kernel_score != score_map_.end());
      assert(!kernel_score->second.score_final);
      assert(kernel_score->second.block_sums.get() != nullptr);

      float sum = 0.0f;
      float* block_sums = kernel_score->second.block_sums.get();
      for (size_t j = 0; j < kBlockSumBufferSize; ++j) {
        sum += block_sums[j];
      }
      kernel_score->second.score = 1.0f - (sum /
          static_cast<float>(vcsmc::kTargetFrameWidthPixels *
                             vcsmc::kFrameHeightPixels));

      // Release temp float storage for savings.
      kernel_score->second.block_sums.reset();
      kernel_score->second.sim_nyuv.reset();
      kernel_score->second.score_final = true;
    }
  }

 private:
  vcsmc::Generation generation_;
  ScoreMap& score_map_;
  MssimScoreStateList& mssim_score_state_list_;
};

class CompeteKernelJob {
 public:
  CompeteKernelJob(
      const vcsmc::Generation generation,
      ScoreMap& score_map,
      vcsmc::TlsPrngList& prng_list,
      size_t tourney_size)
      : generation_(generation),
        score_map_(score_map),
        prng_list_(prng_list),
        tourney_size_(tourney_size) {}
  void operator()(const tbb::blocked_range<size_t>& r) const {
    vcsmc::TlsPrngList::reference engine = prng_list_.local();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      std::shared_ptr<vcsmc::Kernel> kernel = generation_->at(i);

      ScoreMap::iterator kernel_score_it = score_map_.find(kernel);
      assert(kernel_score_it != score_map_.end());
      uint32 victories = 0;
      float kernel_score = kernel_score_it->second.score;

      std::uniform_int_distribution<size_t> tourney_distro(
          0, generation_->size() - 1);
      for (size_t j = 0; j < tourney_size_; ++j) {
        size_t contestant_index = tourney_distro(engine);
        // Lower scores mean better performance.
        ScoreMap::const_iterator competing_kernel = score_map_.find(
            generation_->at(contestant_index));
        if (kernel_score < competing_kernel->second.score)
          ++victories;
      }

      kernel_score_it->second.victories = victories;
    }
  }
 private:
  const vcsmc::Generation generation_;
  ScoreMap& score_map_;
  vcsmc::TlsPrngList& prng_list_;
  size_t tourney_size_;
};

void SaveState(vcsmc::Generation generation,
               const std::shared_ptr<vcsmc::Kernel>& global_minimum,
               const ScoreMap::const_iterator& global_minimum_score) {
  if (FLAGS_generation_output_file != "") {
    if (!vcsmc::SaveGenerationFile(generation, FLAGS_generation_output_file)) {
      fprintf(stderr, "error saving final generation file %s\n",
          FLAGS_generation_output_file.c_str());
    }
  }
  if (FLAGS_image_output_file != "") {
    vcsmc::Image min_sim_image(
        global_minimum_score->second.sim_frame.get() +
        (kLibZ26ImageWidth * kSimSkipLines));
    vcsmc::SaveImage(&min_sim_image, FLAGS_image_output_file);
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

  if (!vcsmc::InitializeCuda(FLAGS_print_stats)) return -1;

  // Read input image, vet, and convert to Nyuv color.
  std::unique_ptr<vcsmc::Image> input_image = vcsmc::LoadImage(
      FLAGS_input_image_file);
  if (!input_image) {
    fprintf(stderr, "error opening input image file %s.\n",
        FLAGS_input_image_file.c_str());
    return -1;
  }
  if (input_image->width() != vcsmc::kTargetFrameWidthPixels ||
      input_image->height() != vcsmc::kFrameHeightPixels) {
    fprintf(stderr, "bad image dimensions on input image file %s.\n",
        FLAGS_input_image_file.c_str());
  }
  std::unique_ptr<float> input_nyuv(new float[vcsmc::kNyuvBufferSize]);
  float* input_nyuv_ptr = input_nyuv.get();
  const uint32* pixel_ptr = input_image->pixels();
  std::memset(input_nyuv_ptr, 0, vcsmc::kNyuvBufferSizeBytes);
  for (size_t i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    for (size_t j = 0; j < vcsmc::kTargetFrameWidthPixels; ++j) {
      vcsmc::RgbaToNormalizedYuv(reinterpret_cast<const uint8*>(pixel_ptr),
          input_nyuv_ptr);
      input_nyuv_ptr[3] = 1.0f;
      ++pixel_ptr;
      input_nyuv_ptr += 4;
    }
    input_nyuv_ptr += (4 * vcsmc::kWindowSize);
  }
  // Copy input Nyuv colors to device, compute mean and stddev asynchronously.
  float4* target_nyuv_device;
  cudaMalloc(&target_nyuv_device, vcsmc::kNyuvBufferSizeBytes);
  float4* target_mean_device;
  cudaMalloc(&target_mean_device, vcsmc::kNyuvBufferSizeBytes);
  float4* target_stddevsq_device;
  cudaMalloc(&target_stddevsq_device, vcsmc::kNyuvBufferSizeBytes);
  cudaMemcpy(target_nyuv_device,
             input_nyuv.get(),
             vcsmc::kNyuvBufferSizeBytes,
             cudaMemcpyHostToDevice);
  dim3 dim_block(16, 16);
  dim3 dim_grid((vcsmc::kTargetFrameWidthPixels / 16) + 1,
                (vcsmc::kFrameHeightPixels / 16) + 1);
  vcsmc::ComputeLocalMean<<<dim_block, dim_grid, 0>>>(target_nyuv_device,
      target_mean_device);
  vcsmc::ComputeLocalStdDevSquared<<<dim_block, dim_grid, 0>>>(
      target_nyuv_device, target_mean_device, target_stddevsq_device);

  vcsmc::SpecList audio_spec_list;
  if (FLAGS_audio_spec_list_file != "") {
    audio_spec_list = vcsmc::ParseSpecListFile(FLAGS_audio_spec_list_file);
    if (!audio_spec_list) {
      fprintf(stderr, "error parsing audio spec list file %s.\n",
          FLAGS_audio_spec_list_file.c_str());
      return -1;
    }
  }

  tbb::task_scheduler_init tbb_init;
  vcsmc::TlsPrngList prng_list;
  MssimScoreStateList mssim_score_state_list;
  vcsmc::Generation generation;
  ScoreMap score_map;

  size_t generation_size = static_cast<size_t>(FLAGS_generation_size);

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
    for (size_t i = 0; i < generation_size; ++i) {
      generation->emplace_back(new vcsmc::Kernel());
    }

    // Generate FLAGS_generation_size number of random individuals.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, generation_size),
        GenerateRandomKernelJob(generation, spec_list, prng_list));
  } else {
    generation = vcsmc::ParseGenerationFile(FLAGS_seed_generation_file);
    if (!generation) {
      fprintf(stderr, "error parsing generation seed file.\n");
      return -1;
    }
    generation_size = generation->size();
    if (audio_spec_list) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, generation_size),
          ClobberSpecJob(generation, audio_spec_list));
    }
  }

  // Initialize simulator global state.
  init_z26_global_tables();

  if (FLAGS_gperf_output_file != "") {
    printf("profiling enabled, saving output to %s.\n",
        FLAGS_gperf_output_file.c_str());
    ProfilerStart(FLAGS_gperf_output_file.c_str());
  }

  // Make sure input image computations are all complete before starting
  // outer sim loop.
  cudaDeviceSynchronize();

  float target_error = strtof(FLAGS_target_error.c_str(), nullptr);
  size_t generation_count = 0;
  size_t streak = 0;
  float last_generation_score = 0.0f;
  bool reroll = false;
  size_t reroll_count = 0;

  uint64 simulation_time_us = 0;
  uint64 simulation_count = 0;

  uint64 scoring_time_us = 0;
  uint64 scoring_count = 0;

  uint64 tourney_time_us = 0;
  uint64 tourney_count = 0;

  uint64 mutate_time_us = 0;
  uint64 mutate_count = 0;

  std::shared_ptr<vcsmc::Kernel> global_minimum;
  ScoreMap::const_iterator global_minimum_score = score_map.end();

  auto epoch_time = std::chrono::high_resolution_clock::now();

  tbb::blocked_range<size_t> front_half(0, generation_size / 2);
  tbb::blocked_range<size_t> back_half(generation_size / 2, generation_size);
  tbb::blocked_range<size_t> full_range(0, generation_size);

  bool full_range_sim_needed = true;

  while (true) {
    if (global_minimum_score != score_map.end() &&
        global_minimum_score->second.score <= target_error) {
      printf("target error reached after %lu generations, terminating.\n",
             generation_count);
      break;
    }

    // Simulate any newly generated kernels to get output color values. The
    // simulation job also launches the CUDA scoring jobs, so they can work in
    // parallel on GPU with the simulation of other Kernels on CPU.
    auto start_of_simulation = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(full_range_sim_needed ? full_range : back_half,
        SimulateKernelJob(generation,
                          target_nyuv_device,
                          target_mean_device,
                          target_stddevsq_device,
                          score_map,
                          mssim_score_state_list));
    simulation_count += full_range_sim_needed ?
        generation_size : generation_size / 2;
    auto end_of_simulation = std::chrono::high_resolution_clock::now();
    simulation_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
        end_of_simulation - start_of_simulation).count();

    // Compute final score for all unscored kernels in the current generation.
    auto start_of_scoring = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(full_range_sim_needed ? full_range : back_half,
        FinalizeScoreJob(generation, score_map, mssim_score_state_list));
    scoring_count += full_range_sim_needed ?
        generation_size : generation_size / 2;
    auto end_of_scoring = std::chrono::high_resolution_clock::now();
    scoring_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
        end_of_scoring - start_of_scoring).count();

    // Conduct tournament based on scores.
    auto start_of_tourney = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(full_range,
        CompeteKernelJob(generation, score_map, prng_list,
        FLAGS_tournament_size));
    auto end_of_tourney = std::chrono::high_resolution_clock::now();
    tourney_count += generation_size;
    tourney_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
        end_of_tourney - start_of_tourney).count();

    // Sort generation by victories.
    std::sort(generation->begin(), generation->end(),
        [score_map](std::shared_ptr<vcsmc::Kernel> a,
                    std::shared_ptr<vcsmc::Kernel> b) {
          ScoreMap::const_iterator a_score = score_map.find(a);
          ScoreMap::const_iterator b_score = score_map.find(b);
          if (a_score->second.victories == b_score->second.victories)
            return a_score->second.score < b_score->second.score;

          return a_score->second.victories > b_score->second.victories;
        });

    if (global_minimum_score == score_map.end() ||
        score_map.find(generation->at(0))->second.score <
        global_minimum_score->second.score) {
      global_minimum = generation->at(0);
      global_minimum_score = score_map.find(generation->at(0));
    }

    assert(global_minimum_score != score_map.end());

    if ((generation_count % FLAGS_save_count) == 0) {
      std::set<uint64> fingerprint_set;
      for (size_t i = 0; i < generation->size(); ++i) {
        fingerprint_set.insert(generation->at(i)->fingerprint());
      }
      double diversity = static_cast<double>(fingerprint_set.size()) /
          static_cast<double>(generation_size);
      auto now = std::chrono::high_resolution_clock::now();
      if (FLAGS_print_stats) {
        printf("gen: %7lu leader: %016" PRIx64 " score: %.8f div: %5.3f "
               "sim: %7" PRIu64 " score: %7" PRIu64 " tourney: %7" PRIu64 " "
               "mutate: %7" PRIu64 " epoch: %7" PRIu64 " elapsed: %7" PRIu64
               "%s\n",
            generation_count,
            generation->at(0)->fingerprint(),
            global_minimum_score->second.score,
            diversity,
            simulation_count ? simulation_count * 1000000 / simulation_time_us
                : 0,
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
      SaveState(generation, global_minimum, global_minimum_score);
      epoch_time = now;
    }

    if (last_generation_score == global_minimum_score->second.score) {
      ++streak;
    } else {
      last_generation_score = global_minimum_score->second.score;
      streak = 0;
    }

    auto start_of_mutate = std::chrono::high_resolution_clock::now();
    if (FLAGS_stagnant_generation_count == 0 ||
        streak < static_cast<size_t>(FLAGS_stagnant_generation_count)) {
      // Replace lowest-scoring half of generation with mutated versions of
      // highest-scoring half of generation.
      for (size_t i = generation_size / 2; i < generation_size; ++i)
        score_map.unsafe_erase(generation->at(i));
      tbb::parallel_for(front_half,
          MutateKernelJob(generation, generation, generation_size / 2, 1,
          score_map, prng_list));
      mutate_count += generation_size / 2;
      full_range_sim_needed = false;
    } else {
      ++reroll_count;
      if (reroll_count > static_cast<size_t>(FLAGS_stagnant_count_limit)) {
        printf("max reroll count reached, terminating.\n");
        break;
      }
      reroll = true;
      streak = 0;
      vcsmc::Generation mutated_generation(
          new std::vector<std::shared_ptr<vcsmc::Kernel>>(generation_size));
      tbb::parallel_for(full_range,
          MutateKernelJob(generation, mutated_generation, 0,
            FLAGS_stagnant_mutation_count, score_map, prng_list));
      generation = mutated_generation;
      mutate_count += generation_size;
      full_range_sim_needed = true;
      score_map.clear();
      global_minimum.reset();
      global_minimum_score = score_map.end();
    }
    auto end_of_mutate = std::chrono::high_resolution_clock::now();
    mutate_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
        end_of_mutate - start_of_mutate).count();

    ++generation_count;
    if (FLAGS_max_generation_number > 0 &&
        generation_count > static_cast<size_t>(FLAGS_max_generation_number)) {
      printf("max generation count reached, terminating.\n");
      break;
    }
  }

  if (FLAGS_gperf_output_file != "") {
    ProfilerStop();
  }

  SaveState(generation, global_minimum, global_minimum_score);

  if (FLAGS_append_kernel_binary != "") {
    vcsmc::AppendKernelBinary(global_minimum, FLAGS_append_kernel_binary);
  }

  return 0;
}
