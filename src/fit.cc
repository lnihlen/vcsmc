// fit.cc - given an input image constructs the minimum-error atari-matched
// color image and computes a fingerprint of it.

#include <cassert>
#include <cinttypes>
#include <farmhash.h>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "tbb/tbb.h"

#include "color.h"
#include "color_table.h"
#include "constants.h"
#include "image.h"
#include "image_file.h"

DEFINE_bool(save_ideal_image, false,
    "If true, fit will save an image file of the ideal color fit image.");

DEFINE_string(image_file, "", "Required image input file path.");
DEFINE_string(output_dir, "", "Required output output index directory.");

typedef std::vector<std::vector<double>> ColorDistances;

class ComputeColorErrorTableJob {
 public:
  ComputeColorErrorTableJob(
      const double* target_lab,
      ColorDistances& distances)
      : target_lab_(target_lab),
        distances_(distances) {}
  void operator()(const tbb::blocked_range<size_t>& r) const {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      const double* atari_lab = vcsmc::kAtariNTSCLabColorTable + (i * 4);
      std::vector<double>& error_table = distances_[i];
      const double* target_lab = target_lab_;
      for (size_t j = 0;
           j < vcsmc::kTargetFrameWidthPixels * vcsmc::kFrameHeightPixels;
           ++j) {
        double error = vcsmc::Ciede2k(target_lab, atari_lab);
        target_lab += 4;
        error_table.push_back(error);
      }
    }
  }
 private:
  const double* target_lab_;
  ColorDistances& distances_;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  tbb::task_scheduler_init tbb_init;

  std::unique_ptr<vcsmc::Image> target_image = vcsmc::LoadImage(
      FLAGS_image_file);
  if (!target_image) {
    fprintf(stderr, "error opening target_image_file \"%s\".\n",
        FLAGS_image_file.c_str());
    return -1;
  }
  if (target_image->width() != vcsmc::kTargetFrameWidthPixels ||
      target_image->height() != vcsmc::kFrameHeightPixels) {
    fprintf(stderr, "Bad image dimensions %dx%d not %dx%d in %s\n",
        target_image->width(),
        target_image->height(),
        vcsmc::kTargetFrameWidthPixels,
        vcsmc::kFrameHeightPixels,
        FLAGS_image_file.c_str());
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

  ColorDistances color_distances(vcsmc::kNTSCColors);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, vcsmc::kNTSCColors),
      ComputeColorErrorTableJob(target_lab.get(), color_distances));

  // Compute theoretical minimum error color table.
  double min_total_error = 0.0;
  std::unique_ptr<uint8[]> ideal_colors(
      new uint8[vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels]);
  uint8* color = ideal_colors.get();
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
    *color = min_pixel_color;
    ++color;
  }

  // Fingerprint ideal image.
  uint64 fingerprint = util::Hash64(
      reinterpret_cast<const char*>(ideal_colors.get()),
        vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels);
  char buf[1024];
  snprintf(buf, 1024, "%s/%016" PRIx64 ".col", FLAGS_output_dir.c_str(),
      fingerprint);
  std::string file_name(buf);
  int fd = open(file_name.c_str(),
      O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd < 0) {
    fprintf(stderr, "error creating file %s.\n", file_name.c_str());
    return -1;
  }
  size_t bytes_written = write(fd, ideal_colors.get(),
      vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels);
  if (bytes_written !=
      vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels) {
    fprintf(stderr, "error writing file %s.\n", file_name.c_str());
    return -1;
  }
  close(fd);

  if (FLAGS_save_ideal_image) {
    vcsmc::Image ideal_image(vcsmc::kTargetFrameWidthPixels,
                             vcsmc::kFrameHeightPixels);
    uint32* pixels = ideal_image.pixels_writeable();
    for (size_t i = 0; i < vcsmc::kFrameSizeBytes; ++i) {
      *pixels = vcsmc::kAtariNTSCABGRColorTable[ideal_colors.get()[i]];
      ++pixels;
      *pixels = vcsmc::kAtariNTSCABGRColorTable[ideal_colors.get()[i]];
      ++pixels;
    }
    snprintf(buf, 1024, "%s/%016" PRIx64 ".png", FLAGS_output_dir.c_str(),
        fingerprint);
    std::string image_file_name(buf);
    vcsmc::SaveImage(&ideal_image, image_file_name);
  }

  printf("%016" PRIx64 "\n", fingerprint);
  return 0;
}
