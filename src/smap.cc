// smap - saliency map generator

#include <string>
#include <vector>

#include "bit_map.h"
#include "cl_buffer.h"
#include "cl_command_queue.h"
#include "cl_device_context.h"
#include "cl_kernel.h"
#include "cl_program.h"
#include "gray_map.h"
#include "types.h"
#include "video_frame_data.h"
#include "video_frame_data_parser.h"

#include "blur_kernel_table.cc"

// With thanks to Sean Aaron Anderson
// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
uint32 NextHighestPower(uint32 v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

std::unique_ptr<vcsmc::BitMap> BuildSaliencyMap(
    vcsmc::CLCommandQueue* queue,
    vcsmc::GrayMap* input_map,
    vcsmc::CLBuffer* blur_kernel_buffer,
    uint32 blur_kernel_size) {
  std::vector<std::unique_ptr<vcsmc::CLKernel>> kernels;
  uint32 width = input_map->width();
  uint32 height = input_map->height();
  uint32 power_width = NextHighestPower(input_map->width());
  uint32 power_height = NextHighestPower(input_map->height());

  // Transfer image to card and unpack into complex float2s and zero-padded
  // out to nearest power of two dimensions.
  std::unique_ptr<vcsmc::CLBuffer> map_buffer =
        vcsmc::CLDeviceContext::MakeBuffer(width * height * sizeof(float));
  map_buffer->EnqueueCopyToDevice(queue, input_map->values());
  std::unique_ptr<vcsmc::CLKernel> unpack_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kUnpackRealToComplex));
  unpack_kernel->SetBufferArgument(0, map_buffer.get());
  unpack_kernel->SetByteArgument(1, sizeof(uint32), &width);
  unpack_kernel->SetByteArgument(2, sizeof(uint32), &height);
  unpack_kernel->SetByteArgument(3, sizeof(uint32), &power_width);
  unpack_kernel->SetByteArgument(4, sizeof(uint32), &power_height);
  std::unique_ptr<vcsmc::CLBuffer> first_fft_buffer =
        vcsmc::CLDeviceContext::MakeBuffer(
            power_width * power_height * 2 * sizeof(float));
  unpack_kernel->SetBufferArgument(5, first_fft_buffer.get());
  unpack_kernel->Enqueue(queue, unpack_kernel->WorkGroupSize());

  std::unique_ptr<vcsmc::CLBuffer> second_fft_buffer =
        vcsmc::CLDeviceContext::MakeBuffer(
            power_width * power_height * 2 * sizeof(float));

  // Forward row FFT
  bool up = true;
  uint32 p = 1;
  while (p < power_width) {
    std::unique_ptr<vcsmc::CLKernel> kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kFFTRadix2));
    kernel->SetBufferArgument(0, up ?
        first_fft_buffer.get() : second_fft_buffer.get());
    kernel->SetByteArgument(1, sizeof(uint32), &p);
    uint32 output_stride = (p == power_width / 2) ? power_height : 1;
    kernel->SetByteArgument(2, sizeof(uint32), &output_stride);
    kernel->SetBufferArgument(3, up ?
        second_fft_buffer.get() : first_fft_buffer.get());
    kernel->Enqueue2D(queue, power_width / 2, power_height);
    kernels.push_back(std::move(kernel));
    up = !up;
    p = p << 1;
  }

  // Forward column FFT
  p = 1;
  while (p < power_height) {
    std::unique_ptr<vcsmc::CLKernel> kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kFFTRadix2));
    kernel->SetBufferArgument(0, up ?
        first_fft_buffer.get() : second_fft_buffer.get());
    kernel->SetByteArgument(1, sizeof(uint32), &p);
    uint32 output_stride = (p == power_height / 2) ? power_width : 1;
    kernel->SetByteArgument(2, sizeof(uint32), &output_stride);
    kernel->SetBufferArgument(3, up ?
        second_fft_buffer.get() : first_fft_buffer.get());
    kernel->Enqueue2D(queue, power_height / 2, power_width);
    kernels.push_back(std::move(kernel));
    up = !up;
    p = p << 1;
  }

  // Spectral Residual
  std::unique_ptr<vcsmc::CLKernel> sr_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kSpectralResidual));
  sr_kernel->SetBufferArgument(0, up ?
      first_fft_buffer.get() : second_fft_buffer.get());
  sr_kernel->SetBufferArgument(1, up ?
      second_fft_buffer.get() : first_fft_buffer.get());
  sr_kernel->Enqueue2D(queue, power_width, power_height);
  up = !up;

  // Inverse row FFT
  std::unique_ptr<vcsmc::CLKernel> ir_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kInverseFFTNormalize));
  ir_kernel->SetBufferArgument(0, up ?
      first_fft_buffer.get() : second_fft_buffer.get());
  float norm = 1.0f;
  ir_kernel->SetByteArgument(1, sizeof(float), &norm);
  ir_kernel->SetBufferArgument(2, up ?
      second_fft_buffer.get() : first_fft_buffer.get());
  ir_kernel->Enqueue2D(queue, power_width / 2, power_height);
  up = !up;
  p = 1;
  while (p < power_width) {
    std::unique_ptr<vcsmc::CLKernel> kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kFFTRadix2));
    kernel->SetBufferArgument(0, up ?
        first_fft_buffer.get() : second_fft_buffer.get());
    kernel->SetByteArgument(1, sizeof(uint32), &p);
    uint32 output_stride = (p == power_width / 2) ? power_height : 1;
    kernel->SetByteArgument(2, sizeof(uint32), &output_stride);
    kernel->SetBufferArgument(3, up ?
        second_fft_buffer.get() : first_fft_buffer.get());
    kernel->Enqueue2D(queue, power_width / 2, power_height);
    kernels.push_back(std::move(kernel));
    up = !up;
    p = p << 1;
  }
  // The data are transposed so we deal with it in transposed rows but still
  // need to normalize with N = width, not height.
  std::unique_ptr<vcsmc::CLKernel> irt_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kInverseFFTNormalize));
  irt_kernel->SetBufferArgument(0, up ?
      first_fft_buffer.get() : second_fft_buffer.get());
  norm = (float)power_width;
  irt_kernel->SetByteArgument(1, sizeof(float), &norm);
  irt_kernel->SetBufferArgument(2, up ?
      second_fft_buffer.get() : first_fft_buffer.get());
  irt_kernel->Enqueue2D(queue, power_height / 2, power_width);
  up = !up;

  // Inverse column FFT
  std::unique_ptr<vcsmc::CLKernel> ic_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kInverseFFTNormalize));
  ic_kernel->SetBufferArgument(0, up ?
      first_fft_buffer.get() : second_fft_buffer.get());
  norm = 1.0f;
  ic_kernel->SetByteArgument(1, sizeof(float), &norm);
  ic_kernel->SetBufferArgument(2, up ?
      second_fft_buffer.get() : first_fft_buffer.get());
  ic_kernel->Enqueue2D(queue, power_height / 2, power_width);
  up = !up;
  p = 1;
  while (p < power_height) {
    std::unique_ptr<vcsmc::CLKernel> kernel(
        vcsmc::CLDeviceContext::MakeKernel(
            vcsmc::CLProgram::Programs::kFFTRadix2));
    kernel->SetBufferArgument(0, up ?
        first_fft_buffer.get() : second_fft_buffer.get());
    kernel->SetByteArgument(1, sizeof(uint32), &p);
    uint32 output_stride = (p == power_height / 2) ? power_width : 1;
    kernel->SetByteArgument(2, sizeof(uint32), &output_stride);
    kernel->SetBufferArgument(3, up ?
        second_fft_buffer.get() : first_fft_buffer.get());
    kernel->Enqueue2D(queue, power_height / 2, power_width);
    kernels.push_back(std::move(kernel));
    up = !up;
    p = p << 1;
  }
  std::unique_ptr<vcsmc::CLKernel> kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kInverseFFTNormalize));
  kernel->SetBufferArgument(0, up ?
      first_fft_buffer.get() : second_fft_buffer.get());
  norm = (float)power_height;
  kernel->SetByteArgument(1, sizeof(float), &norm);
  kernel->SetBufferArgument(2, up ?
      second_fft_buffer.get() : first_fft_buffer.get());
  kernel->Enqueue2D(queue, power_width / 2, power_height);
  up = !up;

  // Drop imaginary values and zero padding.
  std::unique_ptr<vcsmc::CLKernel> pack_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kPackComplexToReal));
  pack_kernel->SetBufferArgument(0,
      up ? first_fft_buffer.get() : second_fft_buffer.get());
  pack_kernel->SetByteArgument(1, sizeof(uint32), &power_width);
  pack_kernel->SetByteArgument(2, sizeof(uint32), &power_height);
  pack_kernel->SetByteArgument(3, sizeof(uint32), &width);
  pack_kernel->SetByteArgument(4, sizeof(uint32), &height);
  pack_kernel->SetBufferArgument(5, map_buffer.get());
  pack_kernel->Enqueue(queue, pack_kernel->WorkGroupSize());

  // Square values to get spectral residual.
  std::unique_ptr<vcsmc::CLKernel> square_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kSquare));
  square_kernel->SetBufferArgument(0, map_buffer.get());
  std::unique_ptr<vcsmc::CLBuffer> square_buffer =
        vcsmc::CLDeviceContext::MakeBuffer(width * height * sizeof(float));
  square_kernel->SetBufferArgument(1, square_buffer.get());
  square_kernel->Enqueue(queue, width * height);

  // Blur results by convolution with Gaussian kernel.
  std::unique_ptr<vcsmc::CLKernel> convolve_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kConvolve));
  convolve_kernel->SetBufferArgument(0, square_buffer.get());
  convolve_kernel->SetBufferArgument(1, blur_kernel_buffer);
  convolve_kernel->SetByteArgument(2, sizeof(uint32), &blur_kernel_size);
  convolve_kernel->SetBufferArgument(3, map_buffer.get());
  convolve_kernel->Enqueue2D(queue, width, height);

  // Compute mean of squared buffer.
  std::unique_ptr<vcsmc::CLKernel> mean_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kMean));
  mean_kernel->SetBufferArgument(0, map_buffer.get());
  uint32 sum_length = width * height;
  mean_kernel->SetByteArgument(1, sizeof(uint32), &sum_length);
  mean_kernel->SetByteArgument(2,
      sizeof(float) * mean_kernel->WorkGroupSize(), NULL);
  std::unique_ptr<vcsmc::CLBuffer> mean_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(sizeof(float)));
  mean_kernel->SetBufferArgument(3, mean_buffer.get());
  mean_kernel->Enqueue(queue, mean_kernel->WorkGroupSize());

  // Compute standard deviation of squared buffer.
  std::unique_ptr<vcsmc::CLKernel> std_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kStandardDeviation));
  std_kernel->SetBufferArgument(0, map_buffer.get());
  std_kernel->SetBufferArgument(1, mean_buffer.get());
  std_kernel->SetByteArgument(2, sizeof(uint32), &sum_length);
  std_kernel->SetByteArgument(3,
      sizeof(float) * std_kernel->WorkGroupSize(), NULL);
  std::unique_ptr<vcsmc::CLBuffer> std_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(sizeof(float)));
  std_kernel->SetBufferArgument(4, std_buffer.get());
  std_kernel->Enqueue(queue, std_kernel->WorkGroupSize());

  std::unique_ptr<vcsmc::CLKernel> bm_kernel(
      vcsmc::CLDeviceContext::MakeKernel(
          vcsmc::CLProgram::Programs::kMakeBitmap));
  bm_kernel->SetBufferArgument(0, map_buffer.get());
  bm_kernel->SetBufferArgument(1, mean_buffer.get());
  bm_kernel->SetBufferArgument(2, std_buffer.get());
  std::unique_ptr<vcsmc::CLBuffer> bm_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(width * height));
  bm_kernel->SetBufferArgument(3, bm_buffer.get());
  bm_kernel->Enqueue(queue, width * height);

  std::unique_ptr<uint8[]> output_bytemap(new uint8[width * height]);
  bm_buffer->EnqueueCopyFromDevice(queue, output_bytemap.get());
  queue->Finish();

  std::unique_ptr<vcsmc::BitMap> bit_map(new vcsmc::BitMap(width, height));
  bit_map->Pack(output_bytemap.get(), width);
  return std::move(bit_map);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr,
        "smap usage:\n"
        "  smap <frame_data.csv> <input_file_spec> <output_file_spec>\n"
        "smap example:\n"
        "  smap frame_data.csv stills_yuv/frame-%%05d.png "
            "stills_smap/frame-%%05d.png\n");
    return -1;
  }

  vcsmc::VideoFrameDataParser parser;
  if (!parser.OpenCSVFile(argv[1])) {
    fprintf(stderr, "error opening ffmpeg frame csv file %s\n", argv[1]);
    return -1;
  }
  std::string input_path_spec(argv[2]);
  std::string output_path_spec(argv[3]);

  if (!vcsmc::CLDeviceContext::Setup()) {
    fprintf(stderr, "OpenCL setup failed!\n");
    return -1;
  }

  const uint32 kMaxFilenameLength = 2048;
  std::unique_ptr<char[]> file_name_buffer(new char[kMaxFilenameLength]);

  std::unique_ptr<vcsmc::CLCommandQueue> queue =
      vcsmc::CLDeviceContext::MakeCommandQueue();
  std::unique_ptr<vcsmc::CLBuffer> blur_kernel_buffer(
      vcsmc::CLDeviceContext::MakeBuffer(
          vcsmc::kBlurKernelSize * vcsmc::kBlurKernelSize * sizeof(float)));
  blur_kernel_buffer->EnqueueCopyToDevice(queue.get(), vcsmc::kBlurKernel);

  std::unique_ptr<vcsmc::VideoFrameDataParser::Frames> frames;
  while (nullptr != (frames = parser.GetNextFrameSet())) {
    // Vector of T BitMaps of XY images:
    //
    //    x ------------------->
    //   +-----------------+
    // y |     t = 0       |
    // | |  +-----------------+
    // | |  |     t = 1       |
    // | |  |                 |
    // v |  |                 |
    //
    std::vector<std::unique_ptr<vcsmc::BitMap>> xy_maps;
    uint32 width = 0;
    uint32 height = 0;

    // Vector of |height| bitmaps of XT images.
    //
    //    x ------------------->
    //   +-----------------+
    // t |     y = 0       |
    // | |  +-----------------+
    // | |  |     y = 1       |
    // | |  |                 |
    // v |  |                 |
    //
    std::vector<std::unique_ptr<vcsmc::GrayMap>> xt_images;

    // Vector of |width| bitmaps of YT images.
    //
    //    y ------------------->
    //   +-----------------+
    // t |     x = 0       |
    // | |  +-----------------+
    // | |  |     x = 1       |
    // | |  |                 |
    // v |  |                 |
    //
    std::vector<std::unique_ptr<vcsmc::GrayMap>> yt_images;

    xy_maps.reserve(frames->size());
    for (uint32 i = 0; i < frames->size(); ++i) {
      vcsmc::VideoFrameData* frame = frames->at(i).get();
      snprintf(file_name_buffer.get(), kMaxFilenameLength,
          input_path_spec.c_str(), frame->frame_number());
      std::string input_file_path(file_name_buffer.get());
      std::unique_ptr<vcsmc::GrayMap> input_map(vcsmc::GrayMap::Load(
          input_file_path));
      if (!input_map) {
        fprintf(stderr, "Error opening %s\n", input_file_path.c_str());
        return -1;
      }

      if (i == 0) {
        width = input_map->width();
        height = input_map->height();
        if (frames->size() > 1) {
          xt_images.reserve(height);
          for (uint32 j = 0; j < height; ++j) {
            xt_images.push_back(std::unique_ptr<vcsmc::GrayMap>(
                new vcsmc::GrayMap(width, frames->size())));
          }

          yt_images.reserve(width);
          for (uint32 j = 0; j < width; ++j) {
            yt_images.push_back(std::unique_ptr<vcsmc::GrayMap>(
                new vcsmc::GrayMap(height, frames->size())));
          }
        }
      } else {
        if (width != input_map->width() ||
            height != input_map->height()) {
          fprintf(stderr, "input size mismatch at frame %llu\n",
              frame->frame_number());
          return -1;
        }
      }

      // If we need to build XT and YT input images from the XY frames then we
      // do so here.
      if (frames->size() > 1) {
        // Copy the j row of the input image into the j XT map at the i row.
        uint32 row_offset = i * width;
        for (uint32 j = 0; j < height; ++j) {
          std::memcpy(xt_images[j]->values_writeable() + row_offset,
              input_map->values() + (j * width), width * sizeof(float));
        }

        // Copy the j column of the input image into j YT map at the i row.
        for (uint32 j = 0; j < width; ++j) {
          const float* map_in = input_map->values() + j;
          float* map_out = yt_images[j]->values_writeable() + (i * height);
          for (uint32 k = 0; k < height; ++k) {
            *map_out = *map_in;
            map_in += width;
            ++map_out;
          }
        }
      }

      std::unique_ptr<vcsmc::BitMap> xy_map = BuildSaliencyMap(
          queue.get(),
          input_map.get(),
          blur_kernel_buffer.get(),
          vcsmc::kBlurKernelSize);

      xy_maps.push_back(std::move(xy_map));
    }

    // If this set of frames contains only a single image then we save it as
    // output and move to the next set of frames.
    if (frames->size() == 1) {
      snprintf(file_name_buffer.get(), kMaxFilenameLength,
          output_path_spec.c_str(), frames->at(0)->frame_number());
      std::string output_file_path(file_name_buffer.get());
      xy_maps[0]->Save(output_file_path);
      continue;
    }

    // Build vector of Y BitMaps of XT images.
    std::vector<std::unique_ptr<vcsmc::BitMap>> xt_maps;
    xt_maps.reserve(height);
    for (uint32 i = 0; i < height; ++i) {
      std::unique_ptr<vcsmc::BitMap> xt_map = BuildSaliencyMap(
          queue.get(),
          xt_images[i].get(),
          blur_kernel_buffer.get(),
          vcsmc::kBlurKernelSize);
      xt_maps.push_back(std::move(xt_map));
    }
    // Can clear the xt_images vector to save memory.
    xt_images.clear();

    // Build vector of X BitMaps of YT images.
    std::vector<std::unique_ptr<vcsmc::BitMap>> yt_maps;
    yt_maps.reserve(width);
    for (uint32 i = 0; i < width; ++i) {
      std::unique_ptr<vcsmc::BitMap> yt_map = BuildSaliencyMap(
          queue.get(),
          yt_images[i].get(),
          blur_kernel_buffer.get(),
          vcsmc::kBlurKernelSize);
      yt_maps.push_back(std::move(yt_map));
    }
    yt_images.clear();

    // Voting and final output.
    for (uint32 i = 0; i < frames->size(); ++i) {
      vcsmc::BitMap vote_map(width, height);
      for (int j = 0; j < height; ++j) {
        for (int k = 0; k < width; ++k) {
          uint32 votes = 0;
          if (xy_maps[i]->bit(k, j))
            ++votes;
          if (xt_maps[j]->bit(k, i))
            ++votes;
          if (yt_maps[k]->bit(j, i))
            ++votes;
          vote_map.SetBit(k, j, votes > 1);
        }
      }
      snprintf(file_name_buffer.get(), kMaxFilenameLength,
          output_path_spec.c_str(), frames->at(i)->frame_number());
      std::string output_file_path(file_name_buffer.get());
      vote_map.Save(output_file_path);
    }
  }

  queue.reset();

  vcsmc::CLDeviceContext::Teardown();
  return 0;
}
