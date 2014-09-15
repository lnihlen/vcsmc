// picc - VCS picture compiler.

#include <cstdio>
#include <string>

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

  // Load input image file.
  vcsmc::TiffImageFile image_file(input_file);
  std::unique_ptr<vcsmc::Image> image(image_file.Load());
  if (!image) {
    fprintf(stderr, "error loading image file: %s\n", input_file.c_str());
    return -1;
  }

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
