// picc - VCS picture compiler.

#include <iostream>

#include "frame.h"
#include "image.h"
#include "kernel.h"
#include "log.h"
#include "switches.h"
#include "tiff_image_file.h"

int main(int argc, char* argv[]) {
  // Keep singleton instances here in main so they will be deconstructed when
  // the program exits.
  vcsmc::Switches switches;
  vcsmc::Log log;

  // Parse command line.
  if (!vcsmc::Switches::Parse(argc, argv)) {
    std::cerr << "picc usage:" << std::endl << std::endl
              << "picc <input_file.tiff> [-o output_file]" << std::endl;
    return -1;
  }

  // Assumes switches have already been parsed, or sadness will occur.
  vcsmc::Log::Setup();

  // Load input image file.
  vcsmc::TiffImageFile image_file(vcsmc::Switches::input_file());
  std::unique_ptr<vcsmc::Image> image(image_file.Load());
  if (!image) {
    std::cerr << "error loading image file: " << vcsmc::Switches::input_file()
              << std::endl;
    return -1;
  }

  // Build Frame from Image.
  std::unique_ptr<vcsmc::Frame> frame(new vcsmc::Frame(image));

  // Kernel takes ownership of frame.
  vcsmc::Kernel kernel(std::move(frame));

  // Fit the frame.
  kernel.Fit();

  // Write the output.
  kernel.Save();

  return 0;
}
