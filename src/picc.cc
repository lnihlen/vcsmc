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
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_image.h"
#include "image.h"
#include "image_file.h"
#include "palette.h"
#include "pixel_strip.h"
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
  // First load saliency map and perform player fitting.
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

  // Now load the color image, and per scanline we cluster into colors depending
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

  std::unique_ptr<vcsmc::CLCommandQueue> queue(
      vcsmc::CLDeviceContext::MakeCommandQueue());

  std::vector<vcsmc::Spec> specs;
  player_0.AppendSpecs(&specs, false);
  player_1.AppendSpecs(&specs, true);

  vcsmc::Random random;
  uint32 pixel_start =
      ((vcsmc::kVSyncScanLines + vcsmc::kVBlankScanLines) *
          vcsmc::kScanLineWidthClocks) + vcsmc::kHBlankWidthClocks;

  for (uint32 i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    std::unique_ptr<vcsmc::PixelStrip> strip = image->GetPixelStrip(i);
    strip->BuildDistances(queue.get());

    // We always try and fit at least 2 colors, one each for the background
    // color and one for the playfield.
    uint32 color_count = 2;
    if (!player_0.IsLineEmpty(i))
      ++color_count;
    if (!player_1.IsLineEmpty(i))
      ++color_count;
    std::unique_ptr<vcsmc::Palette> palette(strip->BuildPalette(color_count,
        &random));

    // Issue background spec for the most numerous color, covering entire line
    // of pixels, and playfield color for the second most frequent.
    uint8 colubk = palette->colu(0);
    uint8 colupf = palette->colu(1);
    specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUBK, colubk,
        vcsmc::Range(pixel_start, pixel_start + vcsmc::kFrameWidthPixels)));
    specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUPF, colupf,
        vcsmc::Range(pixel_start, pixel_start + vcsmc::kFrameWidthPixels)));

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
