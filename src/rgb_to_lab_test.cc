// Testbed for the RGB to Lab Kernel, as well as the OpenCL code.

#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>
#include <OpenCL/opencl.h>
#include <string>
#include <vector>

#include "color_table.cc"

int main(int argc, char* argv[]) {
  // read in program source to a vector of string
  std::string source;
  std::ifstream source_file("cl/rgb_to_lab.cl");
  std::string source_line;
  while (std::getline(source_file, source_line)) {
    source += source_line + "\n";
  }

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

  const char* source_ptr = source.c_str();
  cl_program program = clCreateProgramWithSource(
      context, 1, &source_ptr, NULL, &result);
  if (!program) {
    std::cerr << "failed to create program: " << result << std::endl;
    return -1;
  }

  // Does not have to be blocking, but currently is.
  result = clBuildProgram(program, 0, NULL, "-Werror", NULL, NULL);
  if (result != CL_SUCCESS) {
    size_t len;
    char buffer[4096];
    std::cerr << "failed to build program: " << result << std::endl;
    clGetProgramBuildInfo(
        program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    std::cerr << buffer << std::endl;
    return -1;
  }

  cl_kernel kernel = clCreateKernel(program, "ciede2k", &result);
  if (!kernel || result != CL_SUCCESS) {
    std::cerr << "failed to create compute kernel: " << result << std::endl;
    return -1;
  }

  size_t input_byte_size = sizeof(float) * 4 * data_lines;
  cl_mem lab1_cl = clCreateBuffer(
      context, CL_MEM_READ_ONLY, input_byte_size, NULL, NULL);
  cl_mem lab2_cl = clCreateBuffer(
      context, CL_MEM_READ_ONLY, input_byte_size, NULL, NULL);
  if (!lab1_cl || !lab2_cl) {
    std::cerr << "failed to create input buffers." << std::endl;
    return -1;
  }

  result = clEnqueueWriteBuffer(command_queue,
                                lab1_cl,
                                CL_TRUE,
                                0,
                                input_byte_size,
                                lab1_packed.get(),
                                0,
                                NULL,
                                NULL);
  if (result != CL_SUCCESS) {
    std::cerr << "failed to write lab1_cl" << std::endl;
    return -1;
  }

  result = clEnqueueWriteBuffer(command_queue,
                                lab2_cl,
                                CL_TRUE,
                                0,
                                input_byte_size,
                                lab2_packed.get(),
                                0,
                                NULL,
                                NULL);
  if (result != CL_SUCCESS) {
    std::cerr << "failed to write lab2_cl" << std::endl;
    return -1;
  }

  size_t output_data_size = sizeof(float) * data_lines;
  cl_mem output_cl = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, output_data_size, NULL, NULL);
  if (!output_cl) {
    std::cerr << "failed to create output buffer" << std::endl;
    return -1;
  }

  result = clSetKernelArg(kernel, 0, sizeof(cl_mem), &lab1_cl);
  result |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &lab2_cl);
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
  if (wg_size < data_lines) {
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
  std::unique_ptr<float[]> results_packed(new float[data_lines]);
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
  for (size_t i = 0; i < data_lines; ++i) {
    float diff = fabs((results_packed[i] / expected_results[i]) - 1.0);
    if (diff > 0.0001) {
      std::cerr << "mismatched results on line " << i + 1
                << " expected: " << expected_results[i]
                << " got: " << results_packed[i]
                << std::endl;
    }
  }

  clReleaseMemObject(lab1_cl);
  clReleaseMemObject(lab2_cl);
  clReleaseMemObject(output_cl);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
