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
#include "frame_fitter.h"
#include "image.h"
#include "image_file.h"
#include "random.h"
#include "range.h"
#include "spec.h"
#include "types.h"
#include "video_frame_data.h"
#include "video_frame_data_parser.h"

// If true picc will save mean shift intermediate images into a mean_shift/
// directory from the current working directory.
#define SAVE_MEAN_SHIFT 0

std::unique_ptr<uint8[]> FitColors(const char* image_file_name) {
  std::unique_ptr<vcsmc::Image> image =
      vcsmc::ImageFile::Load(image_file_name);
  if (!image) {
    fprintf(stderr, "error opening image file %s\n", image_file_name);
    return nullptr;
  }

  if (image->height() != vcsmc::kFrameHeightPixels) {
    fprintf(stderr, "unsupported height %d for image file %s\n",
        image->height(), image_file_name);
    return nullptr;
  }

  uint32 image_width = image->width();
  if (image->height() != vcsmc::kFrameHeightPixels)
    return nullptr;

  // Transfer image to card for conversion to Lab color.
  std::unique_ptr<vcsmc::CLCommandQueue> queue(
      vcsmc::CLDeviceContext::MakeCommandQueue());
  std::unique_ptr<vcsmc::CLImage> image_buffer(
      vcsmc::CLDeviceContext::MakeImage(
          image_width, vcsmc::kFrameHeightPixels));
  if (!image_buffer->EnqueueCopyToDevice(queue.get(), image.get())) {
    fprintf(stderr, "error transferring image to OpenCL device.\n");
    return nullptr;
  }

  // Use Mean Shift to segment the image into blobs of color. We then fit
  // each pixel in the shifted image to an optimal background color and issue
  // specs for each contiguous block of color.
  std::unique_ptr<vcsmc::CLBuffer> image_lab(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * 4 * image_width * vcsmc::kFrameHeightPixels));
  std::unique_ptr<vcsmc::CLKernel> lab_kernel(
      vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kRGBToLab));
  lab_kernel->SetImageArgument(0, image_buffer.get());
  lab_kernel->SetBufferArgument(1, image_lab.get());
  lab_kernel->Enqueue2D(queue.get(), image_width, vcsmc::kFrameHeightPixels);

  std::vector<std::unique_ptr<vcsmc::CLKernel>> kernels;

  std::unique_ptr<vcsmc::CLBuffer> image_mean_shift(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * 4 * image_width * vcsmc::kFrameHeightPixels));
  bool forward = true;
  for (uint32 i = 0; i < 4; ++i) {
    std::unique_ptr<vcsmc::CLKernel> mean_shift_kernel(
        vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kMeanShift));
    mean_shift_kernel->SetBufferArgument(0, forward ?
        image_lab.get() : image_mean_shift.get());
    uint32 spatial_bandwidth = 16;
    mean_shift_kernel->SetByteArgument(1, sizeof(uint32), &spatial_bandwidth);
    uint32 range_bandwidth = 16;
    mean_shift_kernel->SetByteArgument(2, sizeof(uint32), &range_bandwidth);
    mean_shift_kernel->SetBufferArgument(3, forward ?
        image_mean_shift.get() : image_lab.get());
    mean_shift_kernel->Enqueue2D(queue.get(), image_width,
        vcsmc::kFrameHeightPixels);
    kernels.push_back(std::move(mean_shift_kernel));
    forward = !forward;
  }

  // If doing an odd number of iterations of mean shift will need to swap
  // buffers, the rest of this code assumes final iteration of mean shift
  // left result in |image_lab|.
  assert(forward);

#if SAVE_MEAN_SHIFT
  std::unique_ptr<vcsmc::CLKernel> rgb_kernel(
      vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kLabToRGB));
  rgb_kernel->SetBufferArgument(0, image_lab.get());
  rgb_kernel->SetImageArgument(1, image_buffer.get());
  rgb_kernel->Enqueue2D(queue.get(), image_width, vcsmc::kFrameHeightPixels);
  vcsmc::Image mean_shift_image(image_width, vcsmc::kFrameHeightPixels);
  image_buffer->EnqueueCopyFromDevice(queue.get(), &mean_shift_image);
  queue->Finish();
  snprintf(file_name_buffer.get(), kMaxFilenameLength,
      "mean_shift/frame-%07llu.png", frame->frame_number());
  vcsmc::ImageFile::Save(&mean_shift_image, file_name_buffer.get());
#endif  // SAVE_MEAN_SHIFT

  // Pull L values from mean shifted Lab color image, then use those to compute
  // saliency map.
  std::unique_ptr<vcsmc::CLBuffer> downsampled_image_lab(
      vcsmc::CLDeviceContext::MakeBuffer(sizeof(float) * 4 *
          vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels));
  std::unique_ptr<vcsmc::CLKernel> downsample_kernel(
      vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kDownsampleColors));
  downsample_kernel->SetBufferArgument(0, image_lab.get());
  downsample_kernel->SetByteArgument(1, sizeof(uint32), &image_width);
  downsample_kernel->SetBufferArgument(2, downsampled_image_lab.get());
  downsample_kernel->Enqueue2D(queue.get(),
      vcsmc::kFrameWidthPixels, vcsmc::kFrameHeightPixels);

  // Upload Atari Lab colors to the card, for calculations on minimum-error
  // colors for each pixel.
  std::unique_ptr<vcsmc::CLBuffer> atari_lab_colors_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * 4 * vcsmc::kNTSCColors));
  atari_lab_colors_buffer->EnqueueCopyToDevice(queue.get(),
      vcsmc::kAtariNTSCLabColorTable);
  std::unique_ptr<vcsmc::CLBuffer> color_errors(
      vcsmc::CLDeviceContext::MakeBuffer(
          sizeof(float) * vcsmc::kFrameWidthPixels * vcsmc::kNTSCColors));
  std::unique_ptr<vcsmc::CLBuffer> colors_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(vcsmc::kFrameWidthPixels));
  std::unique_ptr<uint8[]> colors(
      new uint8[vcsmc::kFrameWidthPixels * vcsmc::kFrameHeightPixels]);
  for (uint32 i = 0; i < vcsmc::kFrameHeightPixels; ++i) {
    // Compute error distances for the ith color.
    std::unique_ptr<vcsmc::CLKernel> ciede_kernel(
        vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kCiede2k));
    ciede_kernel->SetBufferArgument(0, downsampled_image_lab.get());
    ciede_kernel->SetBufferArgument(1, atari_lab_colors_buffer.get());
    ciede_kernel->SetBufferArgument(2, color_errors.get());
    size_t dim[2] = { vcsmc::kFrameWidthPixels, vcsmc::kNTSCColors };
    size_t off[2] = { i * vcsmc::kFrameWidthPixels, 0 };
    ciede_kernel->EnqueueWithOffset(queue.get(), 2, dim, off);

    std::unique_ptr<vcsmc::CLKernel> color_kernel(
        vcsmc::CLDeviceContext::MakeKernel(vcsmc::CLProgram::kMinErrorColor));
    color_kernel->SetBufferArgument(0, color_errors.get());
    uint32 num_colors = vcsmc::kNTSCColors;
    color_kernel->SetByteArgument(1, sizeof(uint32), &num_colors);
    color_kernel->SetBufferArgument(2, colors_buffer.get());
    color_kernel->Enqueue(queue.get(), vcsmc::kFrameWidthPixels);

    colors_buffer->EnqueueCopyFromDevice(queue.get(),
        colors.get() + (i * vcsmc::kFrameWidthPixels));
  }

  queue->Finish();

  return std::move(colors);
}

// If |frame| is a keyframe, FitFrame() will supply colors for |colubk| and
// |colupf|. If it is not, it will simply use the supplied colors.
bool FitFrame(const vcsmc::VideoFrameData* frame,
              const std::string& input_image_path_spec,
              const std::string& output_path_spec) {
  const uint32 kMaxFilenameLength = 2048;
  std::unique_ptr<char[]> file_name_buffer(new char[kMaxFilenameLength]);
  snprintf(file_name_buffer.get(), kMaxFilenameLength,
      input_image_path_spec.c_str(), frame->frame_number());

  std::unique_ptr<uint8[]> colors = FitColors(file_name_buffer.get());
  vcsmc::FrameFitter fit;
  float error = fit.Fit(colors.get());
  printf("fit frame %llu with %f error.\n", frame->frame_number(), error);

  snprintf(file_name_buffer.get(), kMaxFilenameLength, output_path_spec.c_str(),
      frame->frame_number());
  std::unique_ptr<vcsmc::Image> sim = fit.SimulateToImage();
  vcsmc::ImageFile::Save(sim.get(), file_name_buffer.get());
/*
  int bin_fd = open(file_name_buffer.get(), O_WRONLY | O_CREAT | O_TRUNC,
      S_IRUSR | S_IWUSR);
  if (bin_fd < 0) {
    fprintf(stderr, "error opening output bin file %s\n",
        file_name_buffer.get());
    return false;
  }

  write(bin_fd, spec_buffer.get(), spec_buffer_size);
  close(bin_fd);
*/
  return true;
}

int main(int argc, char* argv[]) {
  // Parse command line.
  if (argc != 4) {
    fprintf(stderr,
        "picc usage:\n"
        "  picc <frame_data.csv> <input_image_file_spec> <output_file_spec>\n"
        "picc example:\n"
        "  picc frame_data.csv frames/frame-%%05d.png specs/frame-%%05d.bin\n"
        );
    return -1;
  }

  vcsmc::VideoFrameDataParser parser;
  if (!parser.OpenCSVFile(argv[1])) {
    fprintf(stderr, "error opening ffmpeg frame cvs file %s\n", argv[1]);
    return -1;
  }

  std::string input_image_path_spec(argv[2]);
  std::string output_path_spec(argv[3]);

  if (!vcsmc::CLDeviceContext::Setup()) {
    fprintf(stderr, "OpenCL setup failed!\n");
    return -1;
  }

  std::unique_ptr<vcsmc::VideoFrameDataParser::Frames> frames;

  frames = parser.GetNextFrameSet();
  vcsmc::VideoFrameData* frame = frames->at(0).get();
  FitFrame(frame, input_image_path_spec, output_path_spec);

/*
  while (nullptr != (frames = parser.GetNextFrameSet())) {
    for (uint32 i = 0; i < frames->size(); ++i) {
      vcsmc::VideoFrameData* frame = frames->at(i).get();
      if (!FitFrame(frame, input_image_path_spec, output_path_spec)) {
        return -1;
      }
    }
  }
*/

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
