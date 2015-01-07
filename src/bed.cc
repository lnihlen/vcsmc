// bed - Builds the CIEDE2K error distance table between all Atari colors.

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "cl_kernel.h"
#include "cl_program.h"
#include "color_table.h"
#include "constants.h"

int main(int argc, char* argv[]) {
  if (!vcsmc::CLDeviceContext::Setup()) {
    fprintf(stderr, "OpenCL setup failed!\n");
    return -1;
  }

  std::unique_ptr<vcsmc::CLCommandQueue> queue(
      vcsmc::CLDeviceContext::MakeCommandQueue());

  std::unique_ptr<vcsmc::CLBuffer> atari_lab_colors_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * 4 * vcsmc::kNTSCColors));
  atari_lab_colors_buffer->EnqueueCopyToDevice(queue.get(),
      vcsmc::kAtariNTSCLabColorTable);

  std::unique_ptr<vcsmc::CLBuffer> results_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(sizeof(float) *
          vcsmc::kNTSCColors * vcsmc::kNTSCColors));
  std::unique_ptr<vcsmc::CLKernel> ciede(vcsmc::CLDeviceContext::MakeKernel(
      vcsmc::CLProgram::kCiede2k));
  ciede->SetBufferArgument(0, atari_lab_colors_buffer.get());
  ciede->SetBufferArgument(1, atari_lab_colors_buffer.get());
  ciede->SetBufferArgument(2, results_buffer.get());
  ciede->Enqueue2D(queue.get(), vcsmc::kNTSCColors, vcsmc::kNTSCColors);

  std::unique_ptr<float[]> results(
      new float[vcsmc::kNTSCColors * vcsmc::kNTSCColors]);
  results_buffer->EnqueueCopyFromDevice(queue.get(), results.get());

  queue->Finish();

  int out_fd = open("color_distances_table.cc",
      O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (out_fd < 0) {
    fprintf(stderr, "error opening output spec file %s\n", argv[1]);
    return -1;
  }

  const char kHeaderString[] =
      "// generated file, do not edit!\n"
      "// file generated by bed.cc\n\n"
      "#include \"color_distances_table.h\"\n\n"
      "namespace vcsmc {\n\n"
      "const float kAtariColorDistances[128 * 128] = {\n";

  write(out_fd, kHeaderString, sizeof(kHeaderString) - 1);
  char buffer[128];
  float* e = results.get();
  for (uint32 i = 0;
      i < ((vcsmc::kNTSCColors * vcsmc::kNTSCColors) / 4) - 1; ++i) {
    int len = snprintf(buffer, 128,
        "  %f, %f, %f, %f,\n", e[0], e[1], e[2], e[3]);
    write(out_fd, buffer, len);
    e += 4;
  }

  int len = snprintf(buffer, 128, "  %f, %f, %f, %f\n", e[0], e[1], e[2], e[3]);
  write(out_fd, buffer, len);

  const char kFooterString[] =
      "};\n\n"
      "}  // namespace vcsmc\n";
  write(out_fd, kFooterString, sizeof(kFooterString) - 1);

  close(out_fd);
  return 0;
}
