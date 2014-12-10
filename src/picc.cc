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
#include "player_fitter.h"
#include "random.h"
#include "range.h"
#include "spec.h"
#include "types.h"
#include "video_frame_data.h"
#include "video_frame_data_parser.h"

std::unique_ptr<vcsmc::CLKernel> EnqueuePlayerFit(
    vcsmc::CLCommandQueue* queue,
    vcsmc::CLBuffer* color_errors,
    uint32 start_pixel,
    uint32 pixel_mask,
    vcsmc::CLBuffer* player_color_buffer,
    uint32* player_color) {
  std::unique_ptr<vcsmc::CLKernel> fit_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kFitPlayer));
  fit_kernel->SetBufferArgument(0, color_errors);
  fit_kernel->SetByteArgument(1, sizeof(uint32), &start_pixel);
  fit_kernel->SetByteArgument(2, sizeof(uint32), &pixel_mask);
  uint32 image_width = vcsmc::kFrameWidthPixels;
  fit_kernel->SetByteArgument(3, sizeof(uint32), &image_width);
  uint32 image_height = vcsmc::kFrameHeightPixels;
  fit_kernel->SetByteArgument(4, sizeof(uint32), &image_height);
  fit_kernel->SetByteArgument(5, sizeof(float) * vcsmc::kNTSCColors, nullptr);
  fit_kernel->SetByteArgument(6, sizeof(uint32) * vcsmc::kNTSCColors, nullptr);
  fit_kernel->SetBufferArgument(7, player_color_buffer);
  fit_kernel->Enqueue(queue, vcsmc::kNTSCColors);

  player_color_buffer->EnqueueCopyFromDevice(queue, player_color);
  return std::move(fit_kernel);
}

// If |frame| is a keyframe, FitFrame() will supply colors for |colubk| and
// |colupf|. If it is not, it will simply use the supplied colors.
bool FitFrame(const vcsmc::VideoFrameData* frame,
              const std::string& input_image_path_spec,
              const std::string& input_saliency_map_path_spec,
              const std::string& output_path_spec,
              uint32& colubk,
              uint32& colupf) {
  const uint32 kMaxFilenameLength = 2048;
  std::unique_ptr<char[]> file_name_buffer(new char[kMaxFilenameLength]);
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

  uint32 image_width = image->width();
  assert(image->height() == vcsmc::kFrameHeightPixels);

  // If we are building the whole palette, like on a keyframe, we compute error
  // distance from each Atari color to every pixel in the input image at full
  // image width, then downsample the error buffers to the output size. We
  // therefore need one buffer to hold the intermediate full sized distance
  // results, and kNTSCColors buffers to hold the downsampled results. We still
  // also need the entire structure to compute minimum-error player graphics
  // colors on this line.
  std::unique_ptr<vcsmc::CLBuffer> image_lab(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * 4 * image_width * vcsmc::kFrameHeightPixels));
  std::unique_ptr<vcsmc::CLKernel> lab_kernel(
      vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kRGBToLab));
  lab_kernel->SetImageArgument(0, image->cl_image());
  lab_kernel->SetBufferArgument(1, image_lab.get());
  lab_kernel->Enqueue2D(queue.get(), image_width, vcsmc::kFrameHeightPixels);

  std::unique_ptr<vcsmc::CLBuffer> full_width_errors(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * image_width * vcsmc::kFrameHeightPixels));
  std::unique_ptr<vcsmc::CLBuffer> color_errors(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) *
          vcsmc::kFrameWidthPixels *
          vcsmc::kFrameHeightPixels *
          vcsmc::kNTSCColors));
  std::vector<std::unique_ptr<vcsmc::CLKernel>> kernels;
  for (uint32 i = 0; i < vcsmc::kNTSCColors; ++i) {
    // Compute error distances for the ith color.
    std::unique_ptr<vcsmc::CLKernel> ciede_kernel(
        vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kCiede2k));
    ciede_kernel->SetBufferArgument(0, image_lab.get());
    ciede_kernel->SetByteArgument(1, sizeof(uint32) * 4,
        &(vcsmc::kAtariNTSCLabColorTable[i * 4]));
    ciede_kernel->SetBufferArgument(2, full_width_errors.get());
    ciede_kernel->Enqueue2D(
        queue.get(), image_width, vcsmc::kFrameHeightPixels);
    kernels.push_back(std::move(ciede_kernel));

    // Downsample errors to Atari image width.
    std::unique_ptr<vcsmc::CLKernel> downsample_kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::kDownsampleErrors));
    downsample_kernel->SetBufferArgument(0, full_width_errors.get());
    downsample_kernel->SetByteArgument(1, sizeof(uint32), &image_width);
    uint32 output_offset =
        i * vcsmc::kFrameHeightPixels * vcsmc::kFrameWidthPixels;
    downsample_kernel->SetByteArgument(2, sizeof(uint32), &output_offset);
    downsample_kernel->SetBufferArgument(3, color_errors.get());
    downsample_kernel->Enqueue2D(queue.get(), vcsmc::kFrameWidthPixels,
        vcsmc::kFrameHeightPixels);
  }

  // While distance calculation is cooking on the GPU load saliency map and
  // perform player fitting.
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

  if (frame->is_keyframe()) {
    // Fit image to 2 colors, one for the background and one for the playfield.
    uint32 color_values[2];
    vcsmc::Random random;
    // Generate random initial colors for each of the classes.
    color_values[0] = random.next() % vcsmc::kNTSCColors;
    color_values[1] = random.next() % vcsmc::kNTSCColors;
    std::unique_ptr<vcsmc::CLBuffer> color_values_buffer(
        vcsmc::CLDeviceContext::MakeBuffer(sizeof(uint32) * 2));
    color_values_buffer->EnqueueCopyToDevice(queue.get(), color_values);

    // Since we run k-means on the GPU it is difficult to estimate how many
    // iterations we should run it until the total error becomes stable. We
    // therefore run it in batches, copying the error sums for each iteration
    // within the batch back from the GPU until encounter a stable error value.
    const uint32 kBatchIterations = 8;
    float fit_errors[kBatchIterations];
    std::unique_ptr<vcsmc::CLBuffer> fit_errors_buffer(
        vcsmc::CLDeviceContext::MakeBuffer(sizeof(float) * kBatchIterations));
    std::unique_ptr<vcsmc::CLBuffer> color_classes_buffer(
        vcsmc::CLDeviceContext::MakeBuffer(sizeof(uint32) *
            vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels));
    std::unique_ptr<uint32[]> color_classes(
        new uint32[vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels]);
    const uint32 kMaxIterations = 64;
    uint32 total_iterations = 0;
    bool stable = false;
    uint32 num_colus = 2;
    uint32 frame_width = vcsmc::kFrameWidthPixels;
    uint32 frame_height = vcsmc::kFrameHeightPixels;
    uint32 scratch_size = sizeof(uint32) * vcsmc::kNTSCColors * 2;

    while (!stable && total_iterations < kMaxIterations) {
      for (uint32 i = 0; i < kBatchIterations; ++i) {
        std::unique_ptr<vcsmc::CLKernel> classify(
            vcsmc::CLDeviceContext::MakeKernel(
                vcsmc::CLProgram::Programs::kKMeansClassify));
        classify->SetBufferArgument(0, color_errors.get());
        classify->SetBufferArgument(1, color_values_buffer.get());
        classify->SetByteArgument(2, sizeof(uint32), &num_colus);
        classify->SetBufferArgument(3, color_classes_buffer.get());
        classify->Enqueue2D(queue.get(),
            vcsmc::kFrameWidthPixels, vcsmc::kFrameHeightPixels);
        kernels.push_back(std::move(classify));

        std::unique_ptr<vcsmc::CLKernel> color(vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kKMeansColor));
        color->SetBufferArgument(0, color_errors.get());
        color->SetBufferArgument(1, color_classes_buffer.get());
        color->SetByteArgument(2, sizeof(uint32), &frame_width);
        color->SetByteArgument(3, sizeof(uint32), &frame_height);
        color->SetByteArgument(4, sizeof(uint32), &num_colus);
        color->SetByteArgument(5, sizeof(uint32), &i);
        color->SetByteArgument(6, scratch_size, nullptr);
        color->SetByteArgument(7, scratch_size, nullptr);
        color->SetBufferArgument(8, fit_errors_buffer.get());
        color->SetBufferArgument(9, color_values_buffer.get());
        color->Enqueue(queue.get(), vcsmc::kNTSCColors);
        kernels.push_back(std::move(color));
      }

      fit_errors_buffer->EnqueueCopyFromDevice(queue.get(), fit_errors);
      // Optimistically copy back results as well.
      color_values_buffer->EnqueueCopyFromDevice(queue.get(), color_values);
      color_classes_buffer->EnqueueCopyFromDevice(queue.get(),
          color_classes.get());
      queue->Finish();

      // Check for stable error values within a batch.
      float last_error = fit_errors[0];
      for (uint32 i = 1; i < kBatchIterations; ++i) {
        ++total_iterations;
        if (fit_errors[i] == last_error) {
          stable = true;
          break;
        }
        last_error = fit_errors[i];
      }
    }

    // Histogram classes on CPU.
    uint32 color_counts[2] = { 0, 0 };
    for (uint32 i = 0;
        i < vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels; ++i) {
      assert(color_classes[i] == 0 || color_classes[i] == 1);
      ++color_counts[color_classes[i]];
    }

    uint32 max_color_index = color_counts[0] > color_counts[1] ? 0 : 1;

    colubk = color_values[max_color_index];
    colupf = color_values[1 - max_color_index];
  }

  // Issue background spec for the most numerous color, covering entire frame
  // of pixels, and playfield color for the second most frequent.
  vcsmc::Range entire_frame(
      ((vcsmc::kVSyncScanLines + vcsmc::kVBlankScanLines) *
          vcsmc::kScanLineWidthClocks) + vcsmc::kHBlankWidthClocks,
      ((vcsmc::kVSyncScanLines + vcsmc::kVBlankScanLines +
            vcsmc::kFrameHeightPixels) *
          vcsmc::kScanLineWidthClocks) + 1);
  specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUBK, (uint8)colubk * 2,
      entire_frame));
  specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUPF, (uint8)colupf * 2,
      entire_frame));
  // Always asymmetric playfield for now.
  specs.push_back(vcsmc::Spec(vcsmc::TIA::CTRLPF, 0, entire_frame));

  // Now we have the two colors in question, issue the playfield fitting shader.
  std::unique_ptr<uint32[]> playfield(
      new uint32[vcsmc::kFrameHeightPixels * (vcsmc::kFrameWidthPixels / 4)]);
  std::unique_ptr<vcsmc::CLBuffer> playfield_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(sizeof(uint32) *
          vcsmc::kFrameHeightPixels * (vcsmc::kFrameWidthPixels / 4)));
  std::unique_ptr<vcsmc::CLKernel> pf_fit(vcsmc::CLDeviceContext::MakeKernel(
      vcsmc::CLProgram::Programs::kFitPlayfield));
  pf_fit->SetBufferArgument(0, color_errors.get());
  pf_fit->SetByteArgument(1, sizeof(uint32), &colubk);
  pf_fit->SetByteArgument(2, sizeof(uint32), &colupf);
  pf_fit->SetBufferArgument(3, playfield_buffer.get());
  pf_fit->Enqueue2D(queue.get(), vcsmc::kFrameWidthPixels / 4,
      vcsmc::kFrameHeightPixels);
  playfield_buffer->EnqueueCopyFromDevice(queue.get(), playfield.get());

  queue->Finish();

  std::unique_ptr<vcsmc::CLBuffer> colup0_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(sizeof(uint32)));
  std::unique_ptr<vcsmc::CLBuffer> colup1_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(sizeof(uint32)));
  uint32 pixel_start =
      ((vcsmc::kVSyncScanLines + vcsmc::kVBlankScanLines) *
          vcsmc::kScanLineWidthClocks) + vcsmc::kHBlankWidthClocks;
  uint32* pf_row = playfield.get();
  uint32 row_pixel = 0;
  for (uint32 i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    // Calculate minimum error color for the pixels in the player bitmask, and
    // set a spec for it, if they are drawn.
    uint32 colup0 = 0;
    std::unique_ptr<vcsmc::CLKernel> fit_p0;
    if (!player_0.IsLineEmpty(i)) {
      fit_p0 = EnqueuePlayerFit(queue.get(),
                                color_errors.get(),
                                row_pixel + player_0.row_offset(i),
                                (uint32)player_0.row_mask(i),
                                colup0_buffer.get(),
                                &colup0);
    }

    uint32 colup1 = 0;
    std::unique_ptr<vcsmc::CLKernel> fit_p1;
    if (!player_1.IsLineEmpty(i)) {
      fit_p1 = EnqueuePlayerFit(queue.get(),
                                color_errors.get(),
                                row_pixel + player_1.row_offset(i),
                                (uint32)player_1.row_mask(i),
                                colup1_buffer.get(),
                                &colup1);
    }

    // PF0 D4 through D7 left to right.
    uint8 pf0 = 0;
    for (uint32 j = 0; j < 4; ++j) {
      pf0 = pf0 >> 1;
      if (pf_row[j])
        pf0 = pf0 | 0x80;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF0, pf0,
        vcsmc::Range(pixel_start, pixel_start + 16)));

    // PF1 D7 through D0 left to right.
    uint8 pf1 = 0;
    for (uint32 j = 4; j < 12; ++j) {
      pf1 = pf1 << 1;
      if (pf_row[j])
        pf1 = pf1 | 0x01;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF1, pf1,
        vcsmc::Range(pixel_start + 16, pixel_start + 48)));

    // PF2 D0 through D7 left to right.
    uint8 pf2 = 0;
    for (uint32 j = 12; j < 20; ++j) {
      pf2 = pf2 >> 1;
      if (pf_row[j])
        pf2 = pf2 | 0x80;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF2, pf2,
        vcsmc::Range(pixel_start + 48, pixel_start + 80)));

    // PF0 D4 through D7 left to right.
    pf0 = 0;
    for (uint32 j = 20; j < 24; ++j) {
      pf0 = pf0 >> 1;
      if (pf_row[j])
        pf0 = pf0 | 0x80;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF0, pf0,
        vcsmc::Range(pixel_start + 80, pixel_start + 96)));

    // PF1 D7 through D0 left to right.
    pf1 = 0;
    for (uint32 j = 24; j < 32; ++j) {
      pf1 = pf1 << 1;
      if (pf_row[j])
        pf1 = pf1 | 0x01;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF1, pf1,
        vcsmc::Range(pixel_start + 96, pixel_start + 128)));

    // PF2 D0 through D7 left to right.
    pf2 = 0;
    for (uint32 j = 32; j < 40; ++j) {
      pf2 = pf2 >> 1;
      if (pf_row[j])
        pf2 = pf2 | 0x80;
    }
    specs.push_back(vcsmc::Spec(vcsmc::TIA::PF2, pf2,
        vcsmc::Range(pixel_start + 128, pixel_start + 160)));

    queue->Finish();

    // Issue specs for player color, if needed.
    if (!player_0.IsLineEmpty(i)) {
      uint32 player_start = pixel_start + player_0.row_offset(i);
      specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUP0, (uint8)colup0 * 2,
          vcsmc::Range(player_start, player_start + 8)));
    }

    if (!player_1.IsLineEmpty(i)) {
      uint32 player_start = pixel_start + player_1.row_offset(i);
      specs.push_back(vcsmc::Spec(vcsmc::TIA::COLUP1, (uint8)colup1 * 2,
          vcsmc::Range(player_start, player_start + 8)));
    }

    pixel_start += vcsmc::kScanLineWidthClocks;
    pf_row += vcsmc::kFrameWidthPixels / 4;
    row_pixel += vcsmc::kFrameWidthPixels;
  }

  // All player and playfield specs are now in place. Save to a file and return.
  uint32 spec_buffer_size = 10 * specs.size();
  std::unique_ptr<uint8[]> spec_buffer(new uint8[spec_buffer_size]);
  uint8* buffer_ptr = spec_buffer.get();
  for (uint32 i = 0; i < specs.size(); ++i)
    buffer_ptr += specs[i].Serialize(buffer_ptr);
  snprintf(file_name_buffer.get(), kMaxFilenameLength, output_path_spec.c_str(),
      frame->frame_number());
  int spec_fd = open(file_name_buffer.get(), O_WRONLY | O_CREAT | O_TRUNC,
      S_IRUSR | S_IWUSR);
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
    uint32 colubk = 0;
    uint32 colupf = 0;
    for (uint32 i = 0; i < frames->size(); ++i) {
      vcsmc::VideoFrameData* frame = frames->at(i).get();
      if (!FitFrame(frame, input_image_path_spec, input_saliency_map_path_spec,
          output_path_spec, colubk, colupf)) {
        return -1;
      }
    }
  }

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
