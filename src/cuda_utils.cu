#include "cuda_utils.h"

#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"

namespace vcsmc {

bool InitializeCuda(bool print_stats) {
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
  if (print_stats) {
    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);
    printf("CUDA Device 0: \"%s\" with compute capability %d.%d.\n",
        device_props.name, device_props.major, device_props.minor);
    printf("  total global memory: %lu bytes.\n", device_props.totalGlobalMem);
    printf("  shared memory per blocK: %lu bytes.\n",
        device_props.sharedMemPerBlock);
    printf("  max threads per block: %d threads.\n",
        device_props.maxThreadsPerBlock);
    printf("  total constant memory: %lu bytes.\n", device_props.totalConstMem);
    printf("  can copy and execute simultaneously: %d.\n",
        device_props.deviceOverlap);
    printf("  multiprocessor count: %d.\n", device_props.multiProcessorCount);
  }
  return true;
}

}  // namespace vcsmc
