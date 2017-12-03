#ifndef SRC_CUDA_UTILS_H_
#define SRC_CUDA_UTILS_H_

namespace vcsmc {

// Returns true on success. Will print to stderr on failure.
bool InitializeCuda();

}  // namespace vcsmc

#endif  // SRC_CUDA_UTILS_H_
