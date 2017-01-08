// find_ciede_max_distance - searches RGB 24-bit color space for the maximum
// Ciede2K distance between any two RGB colors. Used for normalizing Ciede2K
// distances to [0, 1] range.

#include <chrono>
#include <stdio.h>

#include "tbb/tbb.h"

#include "color.h"
#include "types.h"

static const uint32 kStartColorABGR = 0x00000000;
static const uint32 kStopColorABGR = 0x00ffffff;
static const uint64 kTotalEvals = (kStopColorABGR - kStartColorABGR) *
  ((kStopColorABGR - kStartColorABGR) / 2);
static const uint32 kStepSize = 255;

struct MaxDistance {
 public:
  MaxDistance() : color_a(0), color_b(0), distance(0) {}

  uint32 color_a;
  uint32 color_b;
  double distance;
};

typedef tbb::combinable<MaxDistance> MaxDistances;

class SearchABGRJob {
 public:
  SearchABGRJob(MaxDistances& max_distances)
    : max_distances_(max_distances) {}

  void operator()(const tbb::blocked_range<uint32>& r) const {
    MaxDistance& local_max_distance = max_distances_.local();
    for (uint32 i = r.begin(); i < r.end(); ++i) {
      double lab_a[4];
      vcsmc::RGBAToLabA(reinterpret_cast<const uint8*>(&i), lab_a);
      for (uint32 j = i + 1; j <= kStopColorABGR; ++j) {
        double lab_b[4];
        vcsmc::RGBAToLabA(reinterpret_cast<const uint8*>(&j), lab_b);
        double distance = vcsmc::Ciede2k(lab_a, lab_b);
        if (distance > local_max_distance.distance) {
          local_max_distance.color_a = i;
          local_max_distance.color_b = j;
          local_max_distance.distance = distance;
        }
      }
    }
  }

 private:
  MaxDistances& max_distances_;
};

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  MaxDistances max_distances;
  uint32 start_range = kStartColorABGR;
  uint32 end_range = start_range + kStepSize;
  uint64 evals = 0;

  auto start_of_run = std::chrono::high_resolution_clock::now();
  while (start_range < kStopColorABGR) {
    tbb::parallel_for(
        tbb::blocked_range<uint32>(start_range, end_range),
        SearchABGRJob(max_distances));
    auto end_of_run = std::chrono::high_resolution_clock::now();
    uint64 time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_of_run - start_of_run).count();
    for (uint32 i = start_range; i < end_range; ++i) {
      evals += kStopColorABGR - i;
    }
    uint64 rate = (evals * 100000) / time_us;
    uint64 eta = (kTotalEvals - evals) / rate;

    printf("rate per second: %llu, eta in seconds: %llu\n", rate, eta);

    start_range = end_range;
    end_range = std::min(start_range + kStepSize, kStopColorABGR + 1);
  }

  MaxDistance md = max_distances.combine(
      [](const MaxDistance& md_a, const MaxDistance& md_b) {
        if (md_a.distance > md_b.distance) return md_a;
        return md_b;
      });

  printf("max distance: %.19g, abgr_a: %x, abgr_b: %x\n",
      md.distance, md.color_a, md.color_b);
  return 0;
}
