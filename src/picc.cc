// picc - VCS picture compiler.

#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "bit_map.h"
#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "cl_kernel.h"
#include "cl_program.h"
#include "color_table.h"
#include "image.h"
#include "image_file.h"
#include "palette.h"
#include "player_fitter.h"
#include "random.h"
#include "range.h"
#include "spec.h"
#include "types.h"
#include "video_frame_data.h"
#include "video_frame_data_parser.h"

uint8 FitPlayerColu(uint32 player_start,
                    const vcsmc::PlayerFitter* player,
                    const vcsmc::PixelStrip* strip,
                    const vcsmc::Palette* palette) {
  uint32 y = strip->row_id();
  // Don't call me on a line with no player graphics.
  assert(!player->IsLineEmpty(y));
  std::unique_ptr<float[]> errors(new float[palette->num_colus()]);
  std::memset(errors.get(), 0, sizeof(float) * palette->num_colus());
  for (uint32 i = 0; i < 8; ++i) {
    if (player->row_mask(y) & (1 << i)) {
      for (uint32 j = 0; j < palette->num_colus(); ++j) {
        errors[j] += strip->Distance(player_start + i, palette->colu(j));
      }
    }
  }

  float min_error = errors[0];
  uint8 min_error_colu = palette->colu(0);
  for (uint32 i = 1; i < palette->num_colus(); ++i) {
    if (errors[i] <= min_error) {
      min_error = errors[i];
      min_error_colu = palette->colu(i);
    }
  }
  return min_error_colu;
}

bool FitFrame(const vcsmc::VideoFrameData* frame,
              const std::string& input_image_path_spec,
              const std::string& input_saliency_map_path_spec,
              const std::string& output_path_spec) {
  // Load the color image, and per scanline we cluster into colors depending
  // on the number of objects rendering on that scene (always BG color with
  // addition of playfield and up to two players), then compute majority color
  // for each object and specify it all.
  snprintf(file_name_buffer.get(), kMaxFilenameLength,
      input_image_path_spec.c_str(), frame->frame_number());
  std::unique_ptr<vcsmc::Image> image =
      vcsmc::ImageFile::Load(file_name_buffer.get());
  if (!image) {
    fprintf(stderr, "error opening image file %s\n", file_name_buffer.get());
    return false;
  }

  if (image->height() != vcsmc::kFrameHeightPixels) {
    fprintf(stderr, "unsupported height %d for image file %s\n",
        image->height(), file_name_buffer.get());
    return false;
  }

  // Transfer image to card for conversion to Lab color.
  std::unique_ptr<vcsmc::CLCommandQueue> queue(
      vcsmc::CLDeviceContext::MakeCommandQueue());
  if (!image->CopyToDevice(queue.get())) {
    fprintf(stderr, "error transferring image to OpenCL device.\n");
    return false;
  }

  assert(image->height() == kFrameHeightPixels);
  std::unique_ptr<vcsmc::CLBuffer> image_lab(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * 4 * image_width * kFrameHeightPixels));
  uint32 image_width = image->width();
  std::unique_ptr<vcsmc::CLKernel> lab_kernel(
      vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kRGBToLab));
  kernel->SetImageArgument(0, image->cl_image());
  kernel->SetBufferArgument(1, image_lab.get());
  kernel->Enqueue2D(queue.get(), image_width, kFrameHeightPixels);

  // While Lab conversion is cooking on the GPU load saliency map and perform
  // player fitting.
  const uint32 kMaxFilenameLength = 2048;
  std::unique_ptr<char[]> file_name_buffer(new char[kMaxFilenameLength]);
  snprintf(file_name_buffer.get(), kMaxFilenameLength,
      input_saliency_map_path_spec.c_str(), frame->frame_number());
  std::unique_ptr<vcsmc::BitMap> saliency_map(
      vcsmc::BitMap::Load(file_name_buffer.get()));
  if (!saliency_map) {
    fprintf(stderr, "error opening saliency map file %s\n",
        file_name_buffer.get());
    return false;
  }

  vcsmc::PlayerFitter player_0;
  player_0.FindOptimumPath(saliency_map.get(), false);
  std::unique_ptr<vcsmc::BitMap> p0_coverage = player_0.MakeCoverageMap();
  saliency_map->Subtract(p0_coverage.get());
  vcsmc::PlayerFitter player_1;
  player_1.FindOptimumPath(saliency_map.get(), true);

  std::vector<vcsmc::Spec> specs;
  player_0.AppendSpecs(&specs, false);
  player_1.AppendSpecs(&specs, true);

  vcsmc::Random random;
  uint32 pixel_start =
      ((vcsmc::kVSyncScanLines + vcsmc::kVBlankScanLines) *
          vcsmc::kScanLineWidthClocks) + vcsmc::kHBlankWidthClocks;

  // Drop image buffer to free up resources on heap and on GPU.
  queue->Finish();
  image.reset();

  // Upload Atari colors in Lab format for error distance calculations.
  std::unique_ptr<vcsmc::CLBuffer> atari_lab_colors(
      vcsmc::CLDeviceContext::MakeBuffer(
        sizeof(float) * 4 * kNTSCColors));
  atari_lab_colors->EnqueueCopyToDevice(queue.get(),
      vcsmc::kAtariNTSCLabColorTable);

  // Reusable buffers for intermediate and final results.
  std::unique_ptr<vcsmc::CLBuffer> full_width_errors(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * image_width * vcsmc::kNTSCColors));
  std::unique_ptr<vcsmc::CLBuffer> downsampled_errors(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * vcsmc::kFrameWidthPixels * vcsmc::kNTSCColors));

  for (uint32 i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    // Compute error distances for the ith row of the image.
    std::unique_ptr<vcsmc::CLKernel> ciede_kernel(
        vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kCiede2k));
    ciede_kernel->SetBufferArgument(0, image_lab.get());
    ciede_kernel->SetBufferArgument(1, atar_lab_colors.get());
    ciede_kernel->SetBufferArgument(2, full_width_errors.get());
    size_t ciede_sizes = { image_width, kNTSCColors };
    size_t ciede_offsets = { i * image_width, 0 };
    ciede_kernel->EnqueueWithOffset(queue.get, 2, ciede_sizes, ciede_offsets);

    // Downsample errors to atari image width.
    std::unique_ptr<vcsmc::CLKernel> downsample_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::kDownsampleErrors));
    downsample_kernel->SetBufferArgument(0, full_width_errors.get());
    downsample_kernel->SetByteArgument(1, sizeof(uint32), &image_width);
    downsample_kernel->SetBufferArgument(2, downsampled_errors.get());
    downsample_kernel->Enqueue2D(queue.get(), vcsmc::kFrameWidthPixels,
        vcsmc::kNTSCColors);

    // We always try and fit at least 2 colors, one each for the background
    // color and one for the playfield.
    uint32 color_count = 2;
    if (!player_0.IsLineEmpty(i))
      ++color_count;
    if (!player_1.IsLineEmpty(i))
      ++color_count;
    std::unique_ptr<vcsmc::Palette> palette(color_count);
    palette->Compute(queue.get(), downsampled_errors.get(), &random);

    // Issue background spec for the most numerous color, covering entire line
    // of pixels, and playfield color for the second most frequent.
    uint8 colubk = palette->colu(0);
    uint8 colupf = palette->colu(1);
    specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUBK, colubk,
        vcsmc::Range(pixel_start, pixel_start + vcsmc::kFrameWidthPixels)));
    specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUPF, colupf,
        vcsmc::Range(pixel_start, pixel_start + vcsmc::kFrameWidthPixels)));

    // TODO: push playfield fitting and player color fitting back to the card?

    // Calculate minimum error color for the pixels in the player bitmask, and
    // set a spec for it.
    if (!player_0.IsLineEmpty(i)) {
      uint32 player_start = pixel_start + player_0.row_offset(i);
      uint8 colup0 = FitPlayerColu(player_start, &player_0, strip.get(),
          palette.get());
      specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUP0, colup0,
          vcsmc::Range(player_start, player_start + 8)));
    }

    if (!player_1.IsLineEmpty(i)) {
      uint32 player_start = pixel_start + player_1.row_offset(i);
      uint8 colup1 = FitPlayerColu(player_start, &player_1, strip.get(),
          palette.get());
      specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUP1, colup1,
          vcsmc::Range(player_start, player_start + 8)));
    }

    // Playfield bits describe color of adjacent 4 pixels for 40 total bits,
    // pixels [0, 4) goes into bit 0, pixels [156, 160) into bit 39.
    uint64 playfield = 0ULL;
    for (uint32 i = 0; i < vcsmc::kFrameWidthPixels; i += 4) {
      float bg_error = strip->Distance(i, colubk);
      bg_error += strip->Distance(i + 1, colubk);
      bg_error += strip->Distance(i + 2, colubk);
      bg_error += strip->Distance(i + 3, colubk);

      float pf_error = strip->Distance(i, colupf);
      pf_error += strip->Distance(i + 1, colupf);
      pf_error += strip->Distance(i + 2, colupf);
      pf_error += strip->Distance(i + 3, colupf);

      if (pf_error < bg_error)
        playfield = playfield | (1ULL << (i / 4));
    }

    // PF0 D4 through D7 left to right.
    uint8 pf0 = 0;
    for (uint32 i = 0; i < 4; ++i) {
      pf0 = pf0 >> 1;
      if (playfield & (1ULL << i))
        pf0 = pf0 | 0x80;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF0, pf0,
        vcsmc::Range(pixel_start, pixel_start + 16)));

    // PF1 D7 through D0 left to right.
    uint8 pf1 = 0;
    for (uint32 i = 4; i < 12; ++i) {
      pf1 = pf1 << 1;
      if (playfield & (1ULL << i))
        pf1 = pf1 | 0x01;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF1, pf1,
        vcsmc::Range(pixel_start + 16, pixel_start + 48)));

    // PF2 D0 through D7 left to right.
    uint8 pf2 = 0;
    for (uint32 i = 12; i < 20; ++i) {
      pf2 = pf2 >> 1;
      if (playfield & (1ULL << i))
        pf2 = pf2 | 0x80;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF2, pf2,
        vcsmc::Range(pixel_start + 48, pixel_start + 80)));

    // PF0 D4 through D7 left to right.
    pf0 = 0;
    for (uint32 i = 20; i < 24; ++i) {
      pf0 = pf0 >> 1;
      if (playfield & (1ULL << i))
        pf0 = pf0 | 0x80;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF0, pf0,
        vcsmc::Range(pixel_start + 80, pixel_start + 96)));

    // PF1 D7 through D0 left to right.
    pf1 = 0;
    for (uint32 i = 24; i < 32; ++i) {
      pf1 = pf1 << 1;
      if (playfield & (1ULL << i))
        pf1 = pf1 | 0x01;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF1, pf1,
        vcsmc::Range(pixel_start + 96, pixel_start + 128)));

    // PF2 D0 through D7 left to right.
    pf2 = 0;
    for (uint32 i = 32; i < 40; ++i) {
      pf2 = pf2 >> 1;
      if (playfield & (1ULL << i))
        pf2 = pf2 | 0x80;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF2, pf2,
        vcsmc::Range(pixel_start + 128, pixel_start + 160)));

    pixel_start += vcsmc::kScanLineWidthClocks;
  }

  // All player and playfield specs are now in place. Save to a file and return.
  uint32 spec_buffer_size = 18 * specs.size();
  std::unique_ptr<uint8[]> spec_buffer(new uint8[spec_buffer_size]);
  uint8* buffer_ptr = spec_buffer.get();
  for (uint32 i = 0; i < specs.size(); ++i)
    buffer_ptr += specs[i].Serialize(buffer_ptr);
  snprintf(file_name_buffer.get(), kMaxFilenameLength, output_path_spec.c_str(),
      frame->frame_number());
  int spec_fd = open(file_name_buffer.get(), O_WRONLY | O_CREAT);
  if (spec_fd < 0) {
    fprintf(stderr, "error opening output spec file %s\n",
        file_name_buffer.get());
    return false;
  }
  write(spec_fd, spec_buffer.get(), spec_buffer_size);
  close(spec_fd);

  return true;
}

int main(int argc, char* argv[]) {
  // Parse command line.
  if (argc != 5) {
    fprintf(stderr,
        "picc usage:\n"
        "  picc <frame_data.csv> <input_image_file_spec> "
            "<input_saliency_map_spec> <output_file_spec>\n"
        "picc example:\n"
        "  picc frame_data.csv frames/frame-%%05d.png smap/frame-%%05d.png "
            "specs/frame-%%05d.spec\n"
        );
    return -1;
  }

  vcsmc::VideoFrameDataParser parser;
  if (!parser.OpenCSVFile(argv[1])) {
    fprintf(stderr, "error opening ffmpeg frame cvs file %s\n", argv[1]);
    return -1;
  }

  std::string input_image_path_spec(argv[2]);
  std::string input_saliency_map_path_spec(argv[3]);
  std::string output_path_spec(argv[4]);

  if (!vcsmc::CLDeviceContext::Setup()) {
    fprintf(stderr, "OpenCL setup failed!\n");
    return -1;
  }

  std::unique_ptr<vcsmc::VideoFrameDataParser::Frames> frames;
  while (nullptr != (frames = parser.GetNextFrameSet())) {
    for (uint32 i = 0; i < frames->size(); ++i) {
      vcsmc::VideoFrameData* frame = frames->at(i).get();
      if (!FitFrame(frame, input_image_path_spec, input_saliency_map_path_spec,
          output_path_spec)) {
        return -1;
      }
    }
  }

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
