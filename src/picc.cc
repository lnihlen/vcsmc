// picc - VCS picture compiler.

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cl_device_context.h"
#include "cl_image.h"
#include "color.h"
#include "image.h"
#include "opcode.h"
#include "schedule.h"
#include "state.h"
#include "tiff_image_file.h"

int main(int argc, char* argv[]) {
  // Parse command line.
  if (argc != 3) {
    printf("picc usage:\n"
           "  picc <input_file_spec.tiff> <frame_rate_in_Hz>\n\n"
           "picc example:\n"
           "  picc test_%%05d.tiff 60\n");
    return -1;
  }
  std::string input_file_spec(argv[1]);
  std::string frame_rate_hz(argv[2]);

  if (!vcsmc::CLDeviceContext::Setup()) {
    printf("OpenCL setup failed, exiting.\n");
    return -1;
  }

  const size_t kFileNameBufferSize = 2048;
  struct stat file_stat;
  int file_counter = 1;
  std::unique_ptr<char[]> file_name(new char[kFileNameBufferSize]);
  snprintf(file_name.get(), kFileNameBufferSize, input_file_spec.c_str(),
      file_counter);
  while (stat(file_name.get(), &file_stat) == 0) {
    // TODO: Multi-threaded fun!
    printf("processing %s\n", file_name.get());

    // Load input image file.
    vcsmc::TiffImageFile image_file(file_name.get());
    std::unique_ptr<vcsmc::Image> image(image_file.Load());
    if (!image) {
      fprintf(stderr, "error loading image file: %s\n", file_name.get());
      return -1;
    }

    ++file_counter;
    snprintf(file_name.get(), kFileNameBufferSize, input_file_spec.c_str(),
        file_counter);
  }

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
