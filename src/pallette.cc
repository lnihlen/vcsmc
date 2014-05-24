#include "pallette.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_set>

#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "constants.h"
#include "pixel_strip.h"
#include "random.h"

namespace vcsmc {

Pallette::Pallette(uint32 num_colus)
    : num_colus_(num_colus),
      error_(0.0f) {
  // A Pallette of zero colors is nonsensical.
  assert(num_colus_ > 0);
}

void Pallette::Compute(const PixelStrip* pixel_strip, Random* random) {
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
  for (size_t i = 0; i < pixel_strip->width(); ++i) {
    uint8 colu_class = classes_[i];
    ++colu_counts[colu_class];
    classes_[i] = colus_[colu_class];
  }

  std::vector<std::pair<uint8, uint8>> colu_sorted;
  colu_sorted.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i)
    colu_sorted.push_back(std::make_pair(colu_counts[i], colus_[i]));
  std::sort(colu_sorted.begin(), colu_sorted.end());

  for (size_t i = 0; i < num_colus_; ++i)
    colus_[i] = colu_sorted[i].second;

  std::cout << std::dec << num_iters << " iterations, final pallette: ";
  for (size_t i = 0; i < num_colus_; ++i)
    std::cout << std::hex << static_cast<uint32>(colus_[i]) << " ";
  std::cout << std::dec << std::endl;
}

float Pallette::Classify(const PixelStrip* pixel_strip) {
  classes_.clear();
  classes_.reserve(pixel_strip->width());
  float total_error = 0.0f;
  for (size_t i = 0; i < pixel_strip->width(); ++i) {
    uint8 colu_class = 0;
    float minimum_error = pixel_strip->distance(i, colus_[0] * 2);
    for (size_t j = 1; j < num_colus_; ++j) {
      float class_error = pixel_strip->distance(i, colus_[j] * 2);
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

void Pallette::Color(const PixelStrip* pixel_strip) {
  std::vector<std::unique_ptr<float[]>> colu_errors;
  colu_errors.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i) {
    std::unique_ptr<float[]> errors(new float[kNTSCColors]);
    memset(errors.get(), 0, sizeof(float) * kNTSCColors);
    colu_errors.push_back(std::move(errors));
  }

  for (size_t i = 0; i < pixel_strip->width(); ++i) {
    uint8 pixel_class = classes_[i];
    for (size_t j = 0; j < kNTSCColors; ++j) {
      colu_errors[pixel_class][j] += pixel_strip->distance(i, j * 2);
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

}  // namespace vcsmc
