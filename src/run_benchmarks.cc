#include "benchmark/benchmark.h"

static void BM_SomeFunction(benchmark::State& state) {
  // Perform setup here
    size_t sum = 0;
  for (auto _ : state) {
    // This code gets timed
      for (size_t i = 0; i < 1000; ++i) {
        sum += i;
      }
  }
}
BENCHMARK(BM_SomeFunction);

// Run all benchmarks.
BENCHMARK_MAIN();

