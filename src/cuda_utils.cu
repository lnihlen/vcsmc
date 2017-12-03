#include "cuda_utils.h"

#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"

namespace vcsmc {

bool InitializeCuda() {
  // Initialize CUDA, use first device.
  int cuda_device_count = 0;
  cudaError_t cuda_error = cudaGetDeviceCount(&cuda_device_count);
  if (cuda_error != cudaSuccess) {
    fprintf(stderr, "CUDA error on device enumeration.\n");
    fprintf(stderr,"%s: %s\n", cudaGetErrorName(cuda_error),
                               cudaGetErrorString(cuda_error));
    return false;
  } else if (!cuda_device_count) {
    fprintf(stderr, "unable to find CUDA device.\n");
    return false;
  }
  // Ensure synchronization behavior consistent with our needs.
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  cudaSetDevice(0);
  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);
  printf("CUDA Device 0: \"%s\" with compute capability %d.%d.\n",
      device_props.name, device_props.major, device_props.minor);
  return true;
}

}  // namespace vcsmc
