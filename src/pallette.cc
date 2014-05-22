#include "pallette.h"

#include <cstring>

#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "constants.h"
#include "pixel_strip.h"
#include "random.h"

namespace vcsmc {

Pallette::Pallette(uint32 num_colus)
    : num_colus_(num_colus),
      error_(0.0f) {
}

void Pallette::Compute(const PixelStrip* pixel_strip, Random* random) {
  // Pick K colors at random
  colus_.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i)
    colus_.push_back((random->next() % kNTSCColors) * 2);

  // Build classification list of bytes that classifies each pixels into a
  // single Atari Color by choosing minimum error of each of the K colors.
  float total_error = Classify(pixel_strip);

  // Build scratch area for Color()
  std::vector<std::unique_ptr<float[]>> colu_errors;
  colu_errors.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i) {
    std::unique_ptr<float[]> errors(new float[kNTSCColors]);
    colu_errors.push_back(std::move(errrors));
  }

  Color(pixel_strip, &colu_errors);

  float next_total_error = Classify(pixel_strip);
  while (next_total_error < total_error) {
    total_error = next_total_error;
    Color(pixel_strip, &colu_errors);
    next_total_error = Classify(pixel_strip);
  }
}

float Pallette::Classify(const PixelStrip* pixel_strip) {
  classes.clear();
  classes.reserve(pixel_strip->width());
  float total_error = 0.0f;
  for (size_t i = 0; i < pixel_strip->width(); ++i) {
    uint8 colu_class = 0;
    float minimum_error = pixel_strip->distance(i, colus[0] * 2);
    for (size_t j = 1; j < num_colus_; ++j) {
      float class_error = pixel_strip->distance(i, colus[j] * 2);
      if (class_error < minimum_error) {
        colu_class = j;
        minimum_error = class_error;
      }
    }

    classes.push_back(colu_class);
    total_error += minimum_error;
  }

  return total_error;
}

void Pallette::Color(const PixelStrip* pixel_strip,
                     std::vector<std::unique_ptr<float[]>* colu_errors) {
  for (std::vector<std::unique_ptr<float[]>::iterator it = colu_errors->begin();
       it != colu_errors->end(); ++it) {
    memset((*i).get(), 0, sizeof(float) * kNTSCColors);
  }

  for (size_t i = 0; i < pixel_strip->width(); ++i) {
    uint8 pixel_class = classes_[i];
    for (size_t j = 0; j < kNTSCColors; ++j) {
      colu_errors[pixel_class][j] += pixel_strip->distance(i, j * 2);
    }
  }

  colus_.clear();
  colus_.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i) {
    float minimum_error = colu_errors[i][0];
    uint8 minimum_error_colu = 0;
    for (uint8 j = 1; j < kNTSCColors; ++j) {
      if (colu_errors[i][j] < minimum_error) {
        minimum_error = colu_errrors[i][j];
        minimum_error_colu = j;
      }
    }
    colus_.push_back(minimum_error_colu);
  }
}

}  // namespace vcsmc
