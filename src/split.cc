// split - being largely a testbed for the development of the VideoDecoder
// functions. Given a movie input, outputs rescaled image frames at 60Hz,
// plus resampled mono audio per-frame blurbs as well.

#include <gflags/gflags.h>
#include <stdio.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#include "Halide.h"
#pragma clang diagnostic pop

#include "constants.h"
#include "image_file.h"
#include "video_decoder.h"

DEFINE_string(movie_input_file, "", "Required path to movie input file.");
DEFINE_string(output_path_spec, "out/frame-\%05lu.png",
              "Output path specification for output files.");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  vcsmc::VideoDecoder::InitDecoderLibrary();

  vcsmc::VideoDecoder decoder;
  if (!decoder.OpenFile(FLAGS_movie_input_file)) {
    fprintf(stderr, "Failed to open file %s, aborting.\n",
        FLAGS_movie_input_file.c_str());
    return -1;
  }

  while (!decoder.AtEndOfFile()) {
    std::unique_ptr<vcsmc::VideoFrame> frame = decoder.GetNextFrame();
    if (!frame) {
      if (!decoder.AtEndOfFile()) fprintf(stderr, "Error decoding frame.\n");
      break;
    }
    printf("frame: %d, key: %d, time: %ld\n",
            frame->frame_number, frame->is_keyframe, frame->frame_time_us);
    char buf[1024];
    snprintf(buf, 1024, FLAGS_output_path_spec.c_str(), frame->frame_number);
    bool save_result = vcsmc::SaveImage(frame->frame_data.begin(),
                                        vcsmc::kTargetFrameWidthPixels,
                                        vcsmc::kFrameHeightPixels,
                                        std::string(buf));
    if (!save_result) {
      fprintf(stderr, "Error saving image file %s\n", buf);
      break;;
    }
  }

  decoder.CloseFile();
  return 0;
}
