#include "colu_strip.h"

#include <cstring>

namespace vcsmc {

ColuStrip::ColuStrip(uint32 row) {
  uint32 start_time = row * kScanLineWidthClocks;
  uint32 end_time = start_time + kScanLineWidthClocks;
  range_.set_end_time(end_time);
  range_.set_start_time(start_time);
  std::memset(colus_, kColuUnpainted, kFrameWidthPixels);
}

}  // namespace vcsmc
