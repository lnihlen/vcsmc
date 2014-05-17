// picc - VCS picture compiler.

#include <iostream>
#include <string>

#include "cl_device_context.h"
//#include "color.h"        // just temp testing of pf_image
//#include "histogram.h"    // ditto
#include "image.h"
#include "kernel.h"
#include "log.h"
#include "opcode.h"
//#include "pixel_strip.h"  // just temp testing of pf_image
#include "scan_line.h"
#include "state.h"
#include "tiff_image_file.h"

int main(int argc, char* argv[]) {
  // Keep singleton instances here in main so they will be deconstructed when
  // the program exits.
  vcsmc::Log log;

  // Parse command line.
  if (argc != 2) {
    std::cerr << "picc usage:" << std::endl
              << "picc <input_file.tiff>" << std::endl;
    return -1;
  }
  std::string input_file(argv[1]);

  // Assumes switches have already been parsed, or sadness will occur.
  vcsmc::Log::Setup();
  if (!vcsmc::CLDeviceContext::Setup()) {
    std::cerr << "OpenCL setup failed, exiting." << std::endl;
    return -1;
  }

/*
  // make test image for pf fitting, 0x00, 0x0e
  std::unique_ptr<vcsmc::Image> test_image(new vcsmc::Image(320, 180));
  std::unique_ptr<vcsmc::PixelStrip> test_strip(new vcsmc::PixelStrip(320));
  for (uint32 i = 0; i < 320; ++i) {
    uint8 colu = (i >> 3) % 2 ? 0x0e : 0x00;
    test_strip->SetPixel(i, vcsmc::Color::AtariColorToABGR(colu));
  }
  for (uint32 i = 0; i < 180; ++i) {
    test_image->SetStrip(i, test_strip.get());
  }
  vcsmc::TiffImageFile test_file("test_pf.tiff");
  test_file.Save(test_image.get());
*/

  // Load input image file.
  vcsmc::TiffImageFile image_file(input_file);
  std::unique_ptr<vcsmc::Image> image(image_file.Load());
  if (!image) {
    std::cerr << "error loading image file: " << input_file << std::endl;
    return -1;
  }

  // Kernel takes ownership of frame.
  vcsmc::Kernel kernel(std::move(image));

  std::cout << "fitting " << input_file << std::endl;

  // Fit the frame.
  kernel.Fit();

  std::cout << "saving fit for " << input_file << ", "
            << kernel.bytes() << " bytes." << std::endl;

  // Write the output.
  kernel.Save();

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
