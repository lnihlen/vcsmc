// picc - VCS picture compiler.

#include <cstdio>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cl_device_context.h"
#include "cl_image.h"
#include "color.h"
#include "image.h"
#include "image_fitter.h"
#include "opcode.h"
#include "spec.h"
#include "state.h"
#include "tiff_image_file.h"

bool ProcessSingleImage(const char* file_name) {
  // Load input image file.
  vcsmc::TiffImageFile image_file(file_name);
  std::unique_ptr<vcsmc::Image> image(image_file.Load());
  if (!image) {
    fprintf(stderr, "error loading image file: %s\n", file_name);
    return false;
  }

  // Run fitter.
  std::unique_ptr<vcsmc::ImageFitter> fitter(
      new vcsmc::ImageFitter(std::move(image)));
  std::unique_ptr<std::vector<vcsmc::Spec>> specs =
      fitter->Fit();

  // Save Specs to file.

  return true;
}

int main(int argc, char* argv[]) {
  // Parse command line.
  if (argc != 4) {
    printf("picc usage:\n"
           "  picc <input_file_spec.tiff> <frame_rate_in_Hz> "
              "<spec_output_file>\n\n"
           "picc example:\n"
           "  picc test_%%05d.tiff 60 demo.spec\n");
    return -1;
  }
  std::string input_file_spec(argv[1]);
  std::string frame_rate_hz(argv[2]);
  std::string spec_output_file_path(argv[3]);

  if (!vcsmc::CLDeviceContext::Setup()) {
    printf("OpenCL setup failed!\n");
    return -1;
  }

  // If the input file spec is not a spec we treat it like a single file,
  // process and return.
  if (input_file_spec.find_first_of('%') == std::string::npos) {
    if (!ProcessSingleImage(input_file_spec.c_str()))
      return -1;
  } else {
    const size_t kFileNameBufferSize = 2048;
    struct stat file_stat;
    int file_counter = 1;
    std::unique_ptr<char[]> file_name(new char[kFileNameBufferSize]);
    snprintf(file_name.get(), kFileNameBufferSize, input_file_spec.c_str(),
        file_counter);
    while (stat(file_name.get(), &file_stat) == 0) {
      // TODO: Multi-threaded fun!
      printf("processing %s\n", file_name.get());

      if (!ProcessSingleImage(file_name.get()))
        return -1;

      ++file_counter;
      snprintf(file_name.get(), kFileNameBufferSize, input_file_spec.c_str(),
          file_counter);
    }
  }

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
