// sim - uses libstella to simulate an output frame image from an input frame
// program.

#include <fcntl.h>
#include <cstring>
#include <gflags/gflags.h>
#include <stdio.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "image.h"
#include "image_file.h"
#include "color_table.h"
#include "types.h"

extern "C" {
#include "libz26/libz26.h"
}

DEFINE_string(frame_binary, "", "The path to frame binary to simulate.");
DEFINE_string(output_image, "", "The path to the output image to generate.");
DEFINE_int32(image_top_offset, 23,
    "The number of lines at the top of the output simulation to skip.");
DEFINE_int32(image_height, 192, "The height of the expected output image.");
DEFINE_string(output_binary, "", "Optional output binary file path, will "
    "contain the entire raw output Atari color codes from the simulator.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  init_z26_global_tables();

  // Read the frame binary file into memory.
  int input_fd = open(FLAGS_frame_binary.c_str(), O_RDONLY);
  if (input_fd < 0) {
    fprintf(stderr, "error opening frame_binary input file %s\n",
        FLAGS_frame_binary.c_str());
    return -1;
  }
  struct stat input_file_stat;
  fstat(input_fd, &input_file_stat);
  size_t input_file_size = input_file_stat.st_size;
  std::unique_ptr<uint8[]> frame_binary_bytes(new uint8[input_file_size]);
  size_t bytes_read = read(input_fd, frame_binary_bytes.get(), input_file_size);
  close(input_fd);
  if (bytes_read != input_file_size) {
    fprintf(stderr, "error reading frame_binary input file %s\n",
        FLAGS_frame_binary.c_str());
    return -1;
  }

  std::unique_ptr<uint8[]> output_picture_bytes(
      new uint8[kLibZ26ImageSizeBytes]);
  memset(output_picture_bytes.get(), 0, kLibZ26ImageSizeBytes);

  simulate_single_frame(frame_binary_bytes.get(), input_file_size,
      output_picture_bytes.get());

  // Palletize output binary into the output image.
  std::unique_ptr<vcsmc::Image> output_image(
      new vcsmc::Image(kLibZ26ImageWidth, FLAGS_image_height));
  uint32* pixel = output_image->pixels_writeable();
  uint8* color_code = output_picture_bytes.get() +
    (kLibZ26ImageWidth * FLAGS_image_top_offset);
  for (int i = 0; i < (kLibZ26ImageWidth * FLAGS_image_height); ++i) {
    *pixel = vcsmc::kAtariNTSCABGRColorTable[*color_code];
    ++pixel;
    ++color_code;
  }

  // Save output image.
  if (!vcsmc::SaveImage(output_image.get(), FLAGS_output_image)) {
    fprintf(stderr, "error saving output image %s\n",
        FLAGS_output_image.c_str());
    return -1;
  }

  // Save output binary if requested for now.
  if (FLAGS_output_binary != "") {
    int output_fd = open(FLAGS_output_binary.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (output_fd < 0) {
      fprintf(stderr, "error opening output_image file %s\n",
          FLAGS_output_image.c_str());
      return -1;
    }
    write(output_fd, output_picture_bytes.get(), kLibZ26ImageSizeBytes);
    close(output_fd);
  }
  return 0;
}
