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

void Pallette::Compute(
    PixelStrip* pixel_strip, CLCommandQueue* queue, Random* random) {
  // Build error distances from each pixel in strip to all Atari colors.
  std::vector<std::unique_ptr<float[]>> error_distances;
  BuildErrorDistances(pixel_strip, queue, &error_distances);

  // Pick K colors at random, keeping them still divided by 2, so [0, 127]
  colus_.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i)
    colus_.push_back(random->next() % kNTSCColors);

  // Build classification list of bytes that classifies each pixels into a
  // single Atari Color by choosing minimum error of each of the K colors.
  float total_error = Classify(error_distances, pixel_strip->width());

  // Build scratch area for Color()
  std::vector<std::unique_ptr<float[]>> colu_errors;
  colu_errors.reserve(num_colus_);
  for (size_t i = 0; i < num_colus_; ++i) {
    std::unique_ptr<float[]> errors(new float[kNTSCColors]);
    colu_errors.push_back(std::move(errrors));
  }

  Color(error_distances, width, &colu_errors);

  float next_total_error = Classify(error_distances, pixel_strip->width());
  while (next_total_error < total_error) {
    total_error = next_total_error;
    Color(error_distances, width, &colu_errors);
    next_total_error = Classify(error_distances, pixel_strip->width());
  }
}

void Pallette::BuildErrorDistances(
    PixelStrip * pixel_strip,
    CLCommandQueue* queue,
    std::vector<std::unique_ptr<float[]>>* error_distances_out) {
  error_distances->reserve(kNTSCColors);

  std::vector<std::unique_ptr<CLBuffer>> error_buffers;
  results_buffers.reserve(kNTSCColors);
  std::vector<std::unique_ptr<CLKernel>> kernels;
  kernels.reserve(kNTSCColors);

  for (size_t i = 0; i < kNTSCColors; ++i) {
    std::unique_ptr<CLBuffer> buffer(CLDeviceContext::MakeBuffer(
        pixel_strip->width() * sizeof(float)));

    std::unique_ptr<CLKernel> kernel(CLDeviceContext::MakeKernel(kCiede2k));
    kernel->SetBufferArgument(0, pixel_strip->lab_strip());
    kernel->SetBufferArgument(1, Color::AtariLabColorBuffer(i * 2));
    kernel->SetBufferArgument(2, buffer.get());
    kernel->Enqueue(queue);

    std::unique_ptr<float[]> errors(new float[pixel_strip->width()]);
    buffer->EnqueueCopyFromDevice(queue, errors.get());

    error_buffers.push_back(std::move(buffer));
    kernels.push_back(std::move(kernel));
    error_distances->push_back(std::move(errors));
  }

  // Block until the math is done and our output buffers are valid.
  queue->Finish();
}

float Pallette::Classify(
    const std::vector<std::unique_ptr<float[]>>& error_distances,
    uint32 width) {
  classes.clear();
  classes.reserve(width);
  float total_error = 0.0f;
  for (size_t i = 0; i < width; ++i) {
    uint8 colu_class = 0;
    float minimum_error = error_distances[colus[0]][i];
    for (size_t j = 1; j < num_colus_; ++j) {
      float class_error = error_distances[colus[j]][i];
      if (class_error < minimum_error) {
        colu_class = j;
        minimum_error = class_error;
      }
    }
    classes_out->push_back(colu_class);
    total_error += minimum_error;
  }

  return total_error;
}

void Pallette::Color(
    const std::vector<std::unique_ptr<float[]>>& error_distances,
    std::vector<std::unique_ptr<float[]>* colu_errors,
    uint32 width) {
  for (std::vector<std::unique_ptr<float[]>::iterator it = colu_errors->begin();
       it != colu_errors->end(); ++it) {
    memset((*i).get(), 0, sizeof(float) * kNTSCColors);
  }

  for (size_t i = 0; i < width; ++i) {
    uint8 pixel_class = classes_[i];
    for (size_t j = 0; j < kNTSCColors; ++j) {
      colu_errors[pixel_class][j] += error_distances[j][i];
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
