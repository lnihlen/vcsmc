#include "benchmark/benchmark.h"

#include "atari_ntsc_lab_color_table.h"
#include "ciede_2k.h"
#include "constants.h"
#include "covariance.h"
#include "gaussian_kernel.h"
#include "mean.h"
#include "rgb_to_lab.h"
#include "ssim.h"
#include "Task.h"
#include "variance.h"

#include "Halide.h"

#include <memory>
#include <random>

static void BM_Ciede2k(benchmark::State& state) {
    Halide::Runtime::Buffer<uint8_t, 3> frameRGB(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);

    for (auto i = 0; i < vcsmc::kFrameSizeBytes * 3; ++i) {
        frameRGB.begin()[i] = distribution(randomEngine);
    }

    // Convert to L*a*b* colors.
    Halide::Runtime::Buffer<float, 3> frameLab(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    rgb_to_lab(frameRGB, frameLab);
    size_t colorOffset = 0;
    Halide::Runtime::Buffer<float, 2> colorDistances(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);

    for (auto _ : state) {
        ciede_2k(frameLab, vcsmc::kAtariNtscLabLColorTable[colorOffset], vcsmc::kAtariNtscLabBColorTable[colorOffset],
            vcsmc::kAtariNtscLabBColorTable[colorOffset], colorDistances);
        colorOffset = (colorOffset + 1) % 128;
    }
}

static void BM_Covariance(benchmark::State& state) {
    Halide::Runtime::Buffer<float, 3> lab1(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    Halide::Runtime::Buffer<float, 2> mean1(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);
    Halide::Runtime::Buffer<float, 3> lab2(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    Halide::Runtime::Buffer<float, 2> mean2(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);
    Halide::Runtime::Buffer<float, 2> kernel = vcsmc::MakeGaussianKernel();
    Halide::Runtime::Buffer<float, 2> covarianceOut(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);

    // Fill Luminance channel with random colors.
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_real_distribution<float> distribution(0, 1.0);
    for (auto i = 0; i < vcsmc::kFrameSizeBytes; ++i) {
        lab1.begin()[i] = distribution(randomEngine);
        lab2.begin()[i] = distribution(randomEngine);
    }

    // Compute means
    mean(lab1, mean1);
    mean(lab2, mean2);

    for (auto _ : state) {
        covariance(lab1, mean1, lab2, mean2, kernel, covarianceOut);
    }
}

static void BM_Mean(benchmark::State& state) {
    Halide::Runtime::Buffer<float, 3> lab(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    Halide::Runtime::Buffer<float, 2> meanOut(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);

    // Fill Luminance channel with random colors.
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_real_distribution<float> distribution(0, 1.0);
    for (auto i = 0; i < vcsmc::kFrameSizeBytes; ++i) {
        lab.begin()[i] = distribution(randomEngine);
    }

    for (auto _ : state) {
        mean(lab, meanOut);
    }
}

static void BM_RgbToLab(benchmark::State& state) {
    Halide::Runtime::Buffer<uint8_t, 3> frameRGB(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);

    for (auto i = 0; i < vcsmc::kFrameSizeBytes * 3; ++i) {
        frameRGB.begin()[i] = distribution(randomEngine);
    }

    // Convert to L*a*b* colors.
    Halide::Runtime::Buffer<float, 3> frameLab(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);

    for (auto _ : state) {
        rgb_to_lab(frameRGB, frameLab);
    }
}

static void BM_Ssim(benchmark::State& state) {
    Halide::Runtime::Buffer<float, 3> lab1(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    Halide::Runtime::Buffer<float, 2> mean1(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);
    Halide::Runtime::Buffer<float, 2> variance1(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);
    Halide::Runtime::Buffer<float, 3> lab2(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    Halide::Runtime::Buffer<float, 2> mean2(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);
    Halide::Runtime::Buffer<float, 2> variance2(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);
    Halide::Runtime::Buffer<float, 2> kernel = vcsmc::MakeGaussianKernel();
    Halide::Runtime::Buffer<float, 2> covarianceOut(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);
    Halide::Runtime::Buffer<float, 2> ssimOut(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);

    // Fill Luminance channel with random colors.
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_real_distribution<float> distribution(0, 1.0);
    for (auto i = 0; i < vcsmc::kFrameSizeBytes; ++i) {
        lab1.begin()[i] = distribution(randomEngine);
        lab2.begin()[i] = distribution(randomEngine);
    }

    mean(lab1, mean1);
    mean(lab2, mean2);
    variance(lab1, mean1, kernel, variance1);
    variance(lab2, mean2, kernel, variance2);
    covariance(lab1, mean1, lab2, mean2, kernel, covarianceOut);

    for (auto _ : state) {
        ssim(mean1, variance1, mean2, variance2, covarianceOut, ssimOut);
    }
}

static void BM_Variance(benchmark::State& state) {
    Halide::Runtime::Buffer<float, 3> lab(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels, 3);
    Halide::Runtime::Buffer<float, 2> meanOut(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);

    // Fill Luminance channel with random colors.
    std::random_device randomDevice;
    std::default_random_engine randomEngine(randomDevice());
    std::uniform_real_distribution<float> distribution(0, 1.0);
    for (auto i = 0; i < vcsmc::kFrameSizeBytes; ++i) {
        lab.begin()[i] = distribution(randomEngine);
    }

    mean(lab, meanOut);

    Halide::Runtime::Buffer<float, 2> kernel = vcsmc::MakeGaussianKernel();
    Halide::Runtime::Buffer<float, 2> varianceOut(vcsmc::kTargetFrameWidthPixels, vcsmc::kFrameHeightPixels);

    for (auto _ : state) {
        variance(lab, meanOut, kernel, varianceOut);
    }
}

static void BM_QuantizeFrames(benchmark::State& state) {
    std::unique_ptr<vcsmc::QuantizeFrames> quantizeFrames(new vcsmc::QuantizeFrames(nullptr));
    quantizeFrames->setupBenchmark();

    for (auto _ : state) {
        quantizeFrames->execute();
    }
}

BENCHMARK(BM_Ciede2k);
BENCHMARK(BM_Covariance);
BENCHMARK(BM_Mean);
BENCHMARK(BM_RgbToLab);
BENCHMARK(BM_Ssim);
BENCHMARK(BM_Variance);
// Multi-threaded algorithms should use real time measurements.
BENCHMARK(BM_QuantizeFrames)->UseRealTime();

BENCHMARK_MAIN();

