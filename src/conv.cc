// conv - extracts an image from the Verilog software TIA simulation output

#include <fcntl.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "color_table.h"
#include "constants.h"
#include "image.h"
#include "image_file.h"
#include "types.h"

int main(int argc, char* argv[]) {
  // Parse command line
  if (argc != 3) {
    fprintf(stderr,
        "conv usage:\n"
        "  conv <sim_output.x> <image_file.png>\n");
    return -1;
  }

  std::string input_sim_file(argv[1]);
  std::string output_image_file(argv[2]);

  const uint32 kSimInputSize =
      2 * vcsmc::kScreenHeight * vcsmc::kScanLineWidthClocks;
  std::unique_ptr<char[]> input_buffer(new char[kSimInputSize]);
  int input_fd = open(input_sim_file.c_str(), O_RDONLY);
  if (input_fd < 0) {
    fprintf(stderr, "error opening file %s", input_sim_file.c_str());
    return -1;
  }
  int bytes_read = read(input_fd, input_buffer.get(), kSimInputSize);
  if (bytes_read < kSimInputSize) {
    fprintf(stderr, "read only %d bytes, expecting %d bytes", bytes_read,
        kSimInputSize);
    return -1;
  }
  close(input_fd);

  vcsmc::Image output_image(vcsmc::kFrameWidthPixels,
      vcsmc::kFrameHeightPixels);
  const char* input_ptr = input_buffer.get() +
      (2 * vcsmc::kScanLineWidthClocks *
          (vcsmc::kVSyncScanLines + vcsmc::kVBlankScanLines));
  uint32* pixel_ptr = output_image.pixels_writeable();
  for (uint32 i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    // Skip HBlank region of scanline.
    input_ptr += 2 * vcsmc::kHBlankWidthClocks;
    for (uint32 j = 0; j < vcsmc::kFrameWidthPixels; ++j) {
      uint8 color_code = 0;
      char nybble = *input_ptr;
      ++input_ptr;
      color_code = ((nybble >= '0' && nybble <= '9') ?
          nybble - '0' : nybble - 'a' + 10) << 4;
      nybble = *input_ptr;
      ++input_ptr;
      color_code |= ((nybble >= '0' && nybble <= '9') ?
          nybble - '0' : nybble - 'a' + 10);
      *pixel_ptr = vcsmc::kAtariNTSCABGRColorTable[color_code / 2];
      ++pixel_ptr;
    }
  }

  if (!vcsmc::ImageFile::Save(&output_image, output_image_file)) {
    fprintf(stderr, "error saving output image file %s",
        output_image_file.c_str());
    return -1;
  }
  return 0;
}
