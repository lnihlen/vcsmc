#include "pallette.h"

#include "cl_command_queue.h"
#include "pixel_strip.h"

namespace vcsmc {

Pallette::Pallette(uint32 num_colus)
    : num_colus_(num_colus) {
}

// For now we compute based on histogram only. Ultimately if the PixelStrip
// has an weight strip we may want to use that to guide our color choices.
void Pallette::Compute(PixelStrip* pixel_strip, CLCommandQueue* queue) {
  // pixel_strip should already have been copied to the GPU.
  
}

}  // namespace vcsmc
