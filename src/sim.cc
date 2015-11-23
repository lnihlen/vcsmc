// sim - uses libstella to simulate an output frame image from an input frame
// program.

#include <gflags/gflags.h>
#include <stdio.h>

DEFINE_string(frame_binary, "", "The path to frame binary to simulate.");
DEFINE_string(output_image, "", "The path to the output image to generate.");
DEFINE_string(cart_format, "vcsmc",
    "What cartridge format to use during emulation. Supported formats are "
    "either \"4k\" or \"vcsmc\".");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  return 0;
}
