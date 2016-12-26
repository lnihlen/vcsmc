// respec - Given an input kernel in yaml, and an audio spec, clobbers the
// existing audio spec and replaces it with provided.

#include <gflags/gflags.h>
#include <stdio.h>
#include <string>

#include "serialization.h"

DEFINE_string(input_kernel_file, "", "Path to kernel file to respec.");
DEFINE_string(audio_spec_file, "", "Path to audio spec file.");
DEFINE_string(output_kernel_file, "", "Path to save kernel yaml");
DEFINE_string(append_kernel_binary, "", "Path to append kernel binary.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  vcsmc::Generation generation = vcsmc::ParseGenerationFile(
      FLAGS_input_kernel_file);
  if (!generation || generation->size() != 1) {
    fprintf(stderr, "error parsing kernel input file %s.\n",
        FLAGS_input_kernel_file.c_str());
    return -1;
  }

  vcsmc::SpecList audio_spec_list;
  audio_spec_list = vcsmc::ParseSpecListFile(FLAGS_audio_spec_file);
  if (!audio_spec_list) {
    fprintf(stderr, "error parsing audio spec file %s.\n",
        FLAGS_audio_spec_file.c_str());
    return -1;
  }

  generation->at(0)->ClobberSpec(audio_spec_list);

  if (FLAGS_output_kernel_file != "") {
    vcsmc::SaveKernelToFile(generation->at(0), FLAGS_output_kernel_file);
  }

  if (FLAGS_append_kernel_binary != "") {
    vcsmc::AppendKernelBinary(generation->at(0), FLAGS_append_kernel_binary);
  }

  return 0;
}
