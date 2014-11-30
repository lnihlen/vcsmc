// exty - Extracts Y plan images from .yuv file as provided by ffmpeg and saves
// them as grayscale images.

#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "gray_map.h"
#include "types.h"

int main(int argc, char* argv[]) {
  // Parse command line.
  if (argc != 5) {
    printf("exty usage:\n"
           "  exty <input_file.yuv> <width> <height> <output_file_spec>\n\n"
           "exty example:\n"
           "  exty slices.yuv 160 192 slices_yuv/frame-y-%%07d.png\n");
    return -1;
  }

  uint32 width = atoi(argv[2]);
  uint32 height = atoi(argv[3]);
  std::string output_file_spec(argv[4]);
  if (width == 0 || height == 0) {
    printf("parsed invalid image dimensions width: %d, height: %d\n", width,
        height);
    return -1;
  }

  // Open the input .yuv file and start to extract frames.
  int yuv_fd = open(argv[1], O_RDONLY);
  if (yuv_fd < 0) {
    printf("unable to open file: %s\n", argv[1]);
    return -1;
  }

  // The .yuv file is unpacked frames, each consisting of the Y plane with
  // luminance at 1 byte/pixel followed by chrominance U and V planes at half
  // resolution. We pull the Y values out and save them as grayscale PNGs, and
  // skip the U and V planes for each frame.
  uint32 bytes_per_image = width * height;
  uint32 bytes_to_skip = (width / 2) * (height / 2) * 2;
  int file_counter = 1;
  std::unique_ptr<uint8[]> image_buffer(new uint8[bytes_per_image]);
  int bytes_read = 0;
  while ((bytes_read = read(yuv_fd, image_buffer.get(), bytes_per_image)) ==
      bytes_per_image) {
    const size_t kFileNameBufferSize = 2048;
    char file_name[kFileNameBufferSize];
    snprintf(file_name, kFileNameBufferSize, output_file_spec.c_str(),
        file_counter);
    vcsmc::ValueMap::SaveFromBytes(file_name, image_buffer.get(), width, height,
        8, width);
    lseek(yuv_fd, bytes_to_skip, SEEK_CUR);
    ++file_counter;
  }

  if (bytes_read > 0) {
    printf("warning: %d bytes left in file\n", bytes_read);
  }

  return 0;
}
