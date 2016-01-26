// pcm - given an input mono wav file generates a series of audio frame spec
// files for blending with kernel generation process.

#include <cassert>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "constants.h"
#include "serialization.h"
#include "sound.h"
#include "sound_file.h"
#include "spec.h"

DEFINE_int32(half_line_offset, 27,
    "Number of cycles to wait per half-line for audio spec.");

DEFINE_string(input_audio_file, "",
    "Required - audio file to process, must be at 31440 Hz mono 32-bit wav.");
DEFINE_string(preview_wav, "",
    "Optional - output preview audio file to save.");
DEFINE_string(output_frame_spec, "",
    "Required - spec string for saving individual frame speclists, "
    "ex: frame-%05lu.spec.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  // Load input audio file.
  std::unique_ptr<vcsmc::Sound> input_wav =
    vcsmc::LoadSound(FLAGS_input_audio_file);
  if (!input_wav) {
    fprintf(stderr, "error loading input audio file %s\n",
        FLAGS_input_audio_file.c_str());
    return -1;
  }

  size_t sample_count = input_wav->number_of_samples();

  // Convert samples to 4-bit.
  std::unique_ptr<uint8[]> pcm_bytes(new uint8[sample_count]);
  for (size_t i = 0; i < input_wav->number_of_samples(); ++i) {
    // Convert into unsigned first.
    uint32 uns = input_wav->samples()[i];
    uns = (uns & 0x7fffffff) | ((~uns) & 0x80000000);
    // Truncate to upper 4 bits.
    pcm_bytes[i] = static_cast<uint8>((uns >> 28) & 0x0000000f);
  }

  size_t frame_count = sample_count / (vcsmc::kScreenHeight * 2);
  uint8* pcm = pcm_bytes.get();
  for (size_t i = 0; i < frame_count; ++i) {
    char buf[1024];
    snprintf(buf, 1024, FLAGS_output_frame_spec.c_str(), i);
    int fd = open(buf, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
      fprintf(stderr, "error creating spec file: %s\n", buf);
      return -1;
    }

    uint32 spec_start = static_cast<uint32>(FLAGS_half_line_offset);
    for (uint32 j = 0; j < vcsmc::kScreenHeight * 2; ++j) {
      size_t chars = snprintf(buf, 1024,
          "- first_cycle: %d\n"
          "  assembler: |\n"
          "    lda #$%x\n"
          "    sta AUDV0\n",
          spec_start, *pcm);
      write(fd, buf, chars);
      ++pcm;
      spec_start += vcsmc::kScanLineWidthCycles / 2;
    }
    close(fd);
  }

  // If requested save the sample output audio.
  if (FLAGS_preview_wav != "") {
    std::unique_ptr<uint32[]> output_samples(new uint32[sample_count]);
    for (size_t i = 0; i < sample_count; ++i) {
      uint32 uns = static_cast<uint32>(pcm_bytes[i]) << 28;
      output_samples[i] = (uns & 0x7fffffff) | ((~uns) & 0x80000000);
    }
    vcsmc::Sound output_sound(std::move(output_samples), sample_count);
    vcsmc::SaveSound(&output_sound, FLAGS_preview_wav);
  }

  return 0;
}