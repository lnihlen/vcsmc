#include "palette.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <unordered_set>

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_kernel.h"
#include "constants.h"
#include "random.h"

namespace vcsmc {

Palette::Palette(uint32 num_colus)
    : num_colus_(num_colus),
      error_(0.0f) {
  // A Palette of zero colors is nonsensical.
  assert(num_colus_ > 0);
}

void Palette::Compute(CLCommandQueue* queue, const CLBuffer* error_buffer,
    Random* random) {
  // We deal in uint32s for the OpenCL data so construct temporary buffers for
  // colors and classes.
  std::unique_ptr<uint32[]> colors_uint32(new uint32[num_colus_]);
  std::unique_ptr<uint32[]> classes_uint32(new uint32[kFrameWidthPixels]);

  // Generate random initial colors for each of the classes.
  for (uint32 i = 0; i < num_colus_; ++i)
    colors_uint32[i] = random->next() % kNTSCColors;

  std::unique_ptr<CLBuffer> colors_buffer(CLDeviceContext::MakeBuffer(
      sizeof(uint32) * num_colus_));
  colors_buffer->EnqueueCopyToDevice(queue, colors_uint32.get());
  std::unique_ptr<CLBuffer> classes_buffer(CLDeviceContext::MakeBuffer(
      sizeof(uint32) * kFrameWidthPixels));

  // Since we run k-means on the GPU it is difficult to estimate how many
  // iterations we should run it until the total error becomes stable. We
  // therefore run it in batches, copying the error sums for each iteration
  // within the batch back from the GPU until encounter a stable error value.
  const uint32 kBatchIterations = 8;
  std::unique_ptr<float[]> fit_errors(new float[kBatchIterations]);
  std::unique_ptr<CLBuffer> fit_errors_buffer(CLDeviceContext::MakeBuffer(
      sizeof(float) * kBatchIterations));

  const uint32 kMaxIterations = 64;
  uint32 total_iterations = 0;
  bool stable = false;
  uint32 image_width = kFrameWidthPixels;
  uint32 scratch_size = sizeof(uint32) * kNTSCColors * num_colus_;

  while (!stable && total_iterations < kMaxIterations) {
    std::vector<std::unique_ptr<CLKernel>> kernels;
    kernels.reserve(kBatchIterations);
    for (uint32 i = 0; i < kBatchIterations; ++i) {
      std::unique_ptr<CLKernel> classify(CLDeviceContext::MakeKernel(
          CLProgram::Programs::kKMeansClassify));
      classify->SetBufferArgument(0, error_buffer);
      classify->SetBufferArgument(1, colors_buffer.get());
      classify->SetByteArgument(2, sizeof(uint32), &num_colus_);
      classify->SetBufferArgument(3, classes_buffer.get());
      classify->Enqueue(queue, kFrameWidthPixels);
      kernels.push_back(std::move(classify));

      std::unique_ptr<CLKernel> color(CLDeviceContext::MakeKernel(
          CLProgram::Programs::kKMeansColor));
      color->SetBufferArgument(0, error_buffer);
      color->SetBufferArgument(1, classes_buffer.get());
      color->SetByteArgument(2, sizeof(uint32), &image_width);
      color->SetByteArgument(3, sizeof(uint32), &num_colus_);
      color->SetByteArgument(4, sizeof(uint32), &i);
      color->SetByteArgument(5, scratch_size, nullptr);
      color->SetByteArgument(6, scratch_size, nullptr);
      color->SetBufferArgument(7, fit_errors_buffer.get());
      color->SetBufferArgument(8, colors_buffer.get());
      color->Enqueue(queue, kNTSCColors);
    }

    fit_errors_buffer->EnqueueCopyFromDevice(queue, fit_errors.get());
    queue->Finish();

    // Check for stable error values.
    float last_error = fit_errors[0];
    for (uint32 i = 1; i < kBatchIterations; ++i) {
      ++total_iterations;
      if (fit_errors[i] == last_error) {
        stable = true;
        error_ = last_error;
        break;
      }
      last_error = fit_errors[i];
    }
  }


  // Compute histogram of classes on GPU.
  std::unique_ptr<CLKernel> histo_kernel(
      CLDeviceContext::MakeKernel(CLProgram::Programs::kHistogramClasses));
  histo_kernel->SetBufferArgument(0, classes_buffer.get());
  histo_kernel->SetByteArgument(1, sizeof(uint32), &num_colus_);
  histo_kernel->SetByteArgument(2,
      sizeof(uint32) * kFrameWidthPixels * num_colus_, nullptr);
  std::unique_ptr<CLBuffer> counts_buffer(
      CLDeviceContext::MakeBuffer(sizeof(uint32) * num_colus_));
  histo_kernel->SetBufferArgument(3, counts_buffer.get());
  histo_kernel->Enqueue(queue, kFrameWidthPixels);

  // Copy classes, colors, and counts back to CPU.
  colors_buffer->EnqueueCopyFromDevice(queue, colors_uint32.get());
  classes_buffer->EnqueueCopyFromDevice(queue, classes_uint32.get());
  std::unique_ptr<uint32[]> counts(new uint32[num_colus_]);
  counts_buffer->EnqueueCopyFromDevice(queue, counts.get());
  queue->Finish();
}

/*
void Palette::Compute(const PixelStrip* pixel_strip, Random* random) {
  // Pick K colors at random
  colus_.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i)
    colus_.push_back(random->next() % kNTSCColors);

  // Build classification list of bytes that classifies each pixels into a
  // single Atari Color by choosing minimum error of each of the K colors.
  error_ = Classify(pixel_strip);

  Color(pixel_strip);

  float next_total_error;
  uint32 stable_count = 0;
  uint32 num_iters = 1;
  do {
    next_total_error = Classify(pixel_strip);
    if (next_total_error == error_) {
      ++stable_count;
    } else {
      error_ = next_total_error;
    }
    Color(pixel_strip);
    ++num_iters;
  } while (stable_count < 10);

  // Replace indices in |classes_| with final stable color values, while at
  // the same time build histogram.
  std::unique_ptr<uint8[]> colu_counts(new uint8[num_colus_]);
  std::memset(colu_counts.get(), 0, num_colus_);
  for (size_t i = 0; i < kFrameWidthPixels; ++i) {
    uint8 colu_class = classes_[i];
    ++colu_counts[colu_class];
    classes_[i] = colus_[colu_class];
  }

  // Sort colus by their counts.
  std::vector<std::pair<uint8, uint8>> colu_sorted;
  colu_sorted.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i)
    colu_sorted.push_back(std::make_pair(colu_counts[i], colus_[i]));
  std::sort(colu_sorted.begin(), colu_sorted.end());

  for (size_t i = 0; i < num_colus_; ++i)
    colus_[i] = colu_sorted[i].second;
}

float Palette::Classify(const PixelStrip* pixel_strip) {
  classes_.clear();
  classes_.reserve(kFrameWidthPixels);
  float total_error = 0.0f;
  for (size_t i = 0; i < kFrameWidthPixels; ++i) {
    uint8 colu_class = 0;
    float minimum_error = pixel_strip->Distance(i, colus_[0] * 2);
    for (size_t j = 1; j < num_colus_; ++j) {
      float class_error = pixel_strip->Distance(i, colus_[j] * 2);
      if (class_error < minimum_error) {
        colu_class = j;
        minimum_error = class_error;
      }
    }

    classes_.push_back(colu_class);
    total_error += minimum_error;
  }

  return total_error;
}

void Palette::Color(const PixelStrip* pixel_strip) {
  std::vector<std::unique_ptr<float[]>> colu_errors;
  colu_errors.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i) {
    std::unique_ptr<float[]> errors(new float[kNTSCColors]);
    memset(errors.get(), 0, sizeof(float) * kNTSCColors);
    colu_errors.push_back(std::move(errors));
  }

  for (size_t i = 0; i < kFrameWidthPixels; ++i) {
    uint8 pixel_class = classes_[i];
    for (size_t j = 0; j < kNTSCColors; ++j) {
      colu_errors[pixel_class][j] += pixel_strip->Distance(i, j * 2);
    }
  }

  colus_.clear();
  colus_.reserve(num_colus_);
  std::unordered_set<uint8> used_colus;
  for (size_t i = 0; i < num_colus_; ++i) {
    float minimum_error = std::numeric_limits<float>::max();
    uint8 minimum_error_colu = 255;
    for (uint8 j = 0; j < kNTSCColors; ++j) {
      if (used_colus.count(j))
        continue;
      if (colu_errors[i][j] < minimum_error) {
        minimum_error = colu_errors[i][j];
        minimum_error_colu = j;
      }
    }
    assert(minimum_error_colu != 255);
    used_colus.insert(minimum_error_colu);
    colus_.push_back(minimum_error_colu);
  }
}
*/
}  // namespace vcsmc
