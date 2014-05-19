// Testbed for the RGB to Lab Kernel, as well as the OpenCL code.

#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>
#include <OpenCL/opencl.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "types.h"
#include "auto/color_table.cc"

int main(int argc, char* argv[]) {
  cl_device_id device_id;
  int result = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (result != CL_SUCCESS) {
    std::cerr << "failed to create device group: " << result << std::endl;
    return -1;
  }

  cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &result);
  if (!context) {
    std::cerr << "failed to create compute context: " << result <<  std::endl;
    return -1;
  }

  cl_command_queue command_queue = clCreateCommandQueue(
      context, device_id, 0, &result);
  if (!command_queue) {
    std::cerr << "failed to create command queue: " << result << std::endl;
    return -1;
  }

  int program_fd = open("cl/rgb_to_lab.cl", O_RDONLY);
    // figure out size to pre-allocate buffer of correct size
  struct stat program_stat;
  if (fstat(program_fd, &program_stat)) {
    close(program_fd);
    return -1;
  }

  std::unique_ptr<char[]> program_bytes(new char[program_stat.st_size + 1]);
  int read_size = read(program_fd, program_bytes.get(), program_stat.st_size);
  close(program_fd);
  if (read_size != program_stat.st_size)
    return -1;

  // NULL-terminate code!
  program_bytes[program_stat.st_size] = '\0';

  const char* source_ptr = program_bytes.get();
  cl_program program = clCreateProgramWithSource(
      context, 1, &source_ptr, NULL, &result);
  if (!program) {
    std::cerr << "failed to create program: " << result << std::endl;
    return -1;
  }

  // Does not have to be blocking, but currently is.
  result = clBuildProgram(program, 0, NULL, "-Werror", NULL, NULL);
  if (result != CL_SUCCESS) {
    std::unique_ptr<char[]> log_char(new char[16384]);
    size_t log_length;
    std::cerr << "failed to build program: " << result << std::endl;
    clGetProgramBuildInfo(program,
                          device_id,
                          CL_PROGRAM_BUILD_LOG,
                          16384,
                          log_char.get(),
                          &log_length);
    std::cerr << log_char.get() << std::endl;
    return -1;
  }

  cl_kernel kernel = clCreateKernel(program, "rgb_to_lab", &result);
  if (!kernel || result != CL_SUCCESS) {
    std::cerr << "failed to create compute kernel: " << result << std::endl;
    return -1;
  }

  // Create single-row image from Atari ABGR color table, as it is a packed
  // little-endian ABGR (CL_RGBA) color row.
  cl_image_format image_format;
  image_format.image_channel_order = CL_RGBA;
  image_format.image_channel_data_type = CL_UNORM_INT8;
  cl_image_desc image_desc;
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  image_desc.image_width = 128;
  image_desc.image_height = 1;
  image_desc.image_depth = 1;  // unused
  image_desc.image_array_size = 1;  // unused
  image_desc.image_row_pitch = 128 * 4;  // row size in bytes or zero
  image_desc.num_mip_levels = 0;  // must be 0
  image_desc.num_samples = 0;  // must be 0
  image_desc.buffer = NULL;  // must be NULL unless buffer type
  cl_mem image = clCreateImage(context,
                               CL_MEM_READ_ONLY,
                               &image_format,
                               &image_desc,
                               NULL,
                               &result);
  if (!image || result != CL_SUCCESS) {
    std::cerr << "failed to create image: " << result << std::endl;
    return -1;
  }

  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {128, 1, 1};
  result = clEnqueueWriteImage(command_queue,
                               image,
                               false,  // reconsider blocking in picc
                               origin,
                               region,
                               128 * 4,
                               1,
                               kAtariNTSCABGRColorTable,
                               0,
                               NULL,
                               NULL);

  if (result != CL_SUCCESS) {
    std::cerr << "failed to write cl_image" << std::endl;
    return -1;
  }

  size_t output_data_size = sizeof(float) * 4 * 128;
  cl_mem output_cl = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, output_data_size, NULL, NULL);
  if (!output_cl) {
    std::cerr << "failed to create output buffer" << std::endl;
    return -1;
  }

  int row = 0;
  result = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image);
  result |= clSetKernelArg(kernel, 1, sizeof(int), &row);
  result |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_cl);
  if (result != CL_SUCCESS) {
    std::cerr << "failed to set kernel argument" << std::endl;
    return -1;
  }

  // get the size of work groups on this device
  size_t wg_size;
  result = clGetKernelWorkGroupInfo(kernel,
                                    device_id,
                                    CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(wg_size),
                                    &wg_size,
                                    NULL);
  if (result != CL_SUCCESS) {
    std::cerr << "unable to get work group size" << std::endl;
    return -1;
  }

  // lazy scheduling for now, we assume work group size is greater than
  // the size of our input set
  if (wg_size < 128) {
    std::cerr << "work group size: " << wg_size << " too small." << std::endl;
    return -1;
  }

  size_t global_size = wg_size;
  result = clEnqueueNDRangeKernel(command_queue,
                                  kernel,
                                  1,
                                  NULL,
                                  &global_size,
                                  &wg_size,
                                  0,
                                  NULL,
                                  NULL);
  if (result != CL_SUCCESS) {
    std::cerr << "failed to execute kernel." << std::endl;
    return -1;
  }

  // Wait for queue to empty.
  clFinish(command_queue);

  // Read back results for verification.
  std::unique_ptr<float[]> results_packed(new float[128 * 4]);
  result = clEnqueueReadBuffer(command_queue,
                               output_cl,
                               CL_TRUE,
                               0,
                               output_data_size,
                               results_packed.get(),
                               0,
                               NULL,
                               NULL);
  if (result != CL_SUCCESS) {
    std::cerr << "failed to read back results" << std::endl;
    return -1;
  }

  // Compare results.
  for (size_t i = 0; i < 128 * 4; ++i) {
    float diff = fabs((results_packed[i] / kAtariNTSCLabColorTable[i]) - 1.0);
    if (diff > 0.0001) {
      std::cerr << "mismatched results in element " << i
                << " expected: " << kAtariNTSCLabColorTable[i]
                << " got: " << results_packed[i]
                << std::endl;
    }
  }

  clReleaseMemObject(image);
  clReleaseMemObject(output_cl);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
