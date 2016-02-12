// build_error_distances - generates color_distance_table.h/cc in the provided
// output directory. Computes all distances from all Atari colors to all other
// Atari colors.

#include <fcntl.h>
#include <gflags/gflags.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "color.h"
#include "color_table.h"
#include "constants.h"

DEFINE_string(output_directory, "../out",
    "Output directory to save generated files to.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  // Save code file as we compute the distances.
  std::string code_path = FLAGS_output_directory + "/" +
      "color_distance_table.cc";
  int fd = open(code_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
      S_IRUSR | S_IWUSR);
  if (fd < 0) {
    fprintf(stderr, "error creating code file %s.\n", code_path.c_str());
    return -1;
  }
  std::string code_top_string =
      "// File generated by src/build_error_distances.cc, edit there.\n\n"

      "#include \"color_distance_table.h\"\n\n"

      "namespace vcsmc {\n\n"

      "const double kColorDistanceNTSC[128*128] = {\n\n";

  size_t bytes_written = write(fd, code_top_string.c_str(),
      code_top_string.size());
  if (bytes_written != code_top_string.size()) {
    fprintf(stderr, "error writing code file %s.\n", code_path.c_str());
    return -1;
  }

  char buf[1024];
  for (size_t i = 0; i < vcsmc::kNTSCColors; ++i) {
    const double* i_lab = vcsmc::kAtariNTSCLabColorTable + (i * 4);
    const double* j_lab = vcsmc::kAtariNTSCLabColorTable;
    for (size_t j = 0; j < vcsmc::kNTSCColors; j += 2) {
      size_t len = snprintf(buf, 1024, "%3.16f, %3.16f,\n",
        vcsmc::Ciede2k(i_lab, j_lab),
        vcsmc::Ciede2k(i_lab, j_lab + 4));
      j_lab += 8;
      write(fd, buf, len);
    }
  }

  std::string code_bottom_string =
      "\n};\n\n}  // namespace vcsmc\n";
  bytes_written = write(fd, code_bottom_string.c_str(),
      code_bottom_string.size());
  close(fd);
  return 0;
}