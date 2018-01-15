// build_nyuv_table - generates atari_ntsc_nyuv_color_table.h/cc in the provided
// output directory. Converts RGBA Atari colors to NYUV colors.

#include <fcntl.h>
#include <gflags/gflags.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "atari_ntsc_abgr_color_table.h"
#include "color.h"
#include "constants.h"

DEFINE_string(output_directory, "../out",
    "Output directory to save generated files to.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  // Build NYUV table from RGBA table.
  std::unique_ptr<float> nyuv_table(new float[128*4]);
  float* nyuv = nyuv_table.get();
  for (size_t i = 0; i < 128; ++i) {
    vcsmc::RgbaToNormalizedYuv(
        reinterpret_cast<const uint8*>(vcsmc::kAtariNtscAbgrColorTable + i),
        nyuv);
    nyuv += 4;
  }

  // Save code file of converted colors.
  std::string code_path = FLAGS_output_directory + "/" +
      "atari_ntsc_nyuv_color_table.cc";
  int fd = open(code_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
      S_IRUSR | S_IWUSR);
  if (fd < 0) {
    fprintf(stderr, "error creating code file %s.\n", code_path.c_str());
    return -1;
  }
  std::string code_top_string =
      "// File generated by src/build_nyuv_table.cc, edit there.\n\n"

      "#include \"atari_ntsc_nyuv_color_table.h\"\n\n"

      "namespace vcsmc {\n\n"

      "const float kAtariNtscNyuvColorTable[128*4] = {\n";

  size_t bytes_written = write(fd, code_top_string.c_str(),
      code_top_string.size());
  if (bytes_written != code_top_string.size()) {
    fprintf(stderr, "error writing code file %s.\n", code_path.c_str());
    return -1;
  }

  char buf[1024];
  nyuv = nyuv_table.get();
  for (size_t i = 0; i < 128; ++i) {
    size_t len = snprintf(buf, 1024, "  %.12g, %.12g, %.12g, 1.0,\n",
        nyuv[0], nyuv[1], nyuv[2]);
    nyuv += 4;
    write(fd, buf, len);
  }

  std::string code_bottom_string =
      "\n};\n\n}  // namespace vcsmc\n";
  bytes_written = write(fd, code_bottom_string.c_str(),
      code_bottom_string.size());
  close(fd);

  return 0;
}