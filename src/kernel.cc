#include "kernel.h"

namespace vcsmc {

Kernel::Kernel(std::unique_ptr<Frame> target_frame)
    : target_frame_(std::move(target_frame)) {
}

void Kernel::Fit() {

}

void Kernel::Save() {
}

}  // namespace vcsmc
