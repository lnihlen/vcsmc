#include "frame_fitter.h"

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

#include "color_distances_table.h"
#include "color_table.h"
#include "image.h"
#include "line_fitter.h"
#include "opcode.h"
#include "random.h"
#include "state.h"

namespace vcsmc {

FrameFitter::FrameFitter() {
}

FrameFitter::~FrameFitter() {
}

float FrameFitter::Fit(const uint8* half_colus) {
  AppendFramePrefixOpCodes();
  uint32 seed[] = {
      0xcc02f8c5, 0xa0997c34, 0xa86303df, 0xac1d3f0d,
      0x2508f747, 0xec381a4f, 0x65e47935, 0xddbc2287,
      0x29e1672b, 0xf6a6bed3, 0x27b13937, 0xbce6348c
  };
  Random random(seed);
  float error = 0.0f;
  for (uint32 i = 0; i < kFrameHeightPixels + 1; ++i) {
    printf("fitting line %d of %d..\n", i, kFrameHeightPixels);
    uint32 scan_line = kVSyncScanLines + kVBlankScanLines + i - 1;
    const State* entry_state = states_.rbegin()->get();
//    printf("%d == %d\n", scan_line * kScanLineWidthClocks,
//        entry_state->range().start_time());
    assert(scan_line * kScanLineWidthClocks ==
        entry_state->range().start_time());
    LineFitter line_fitter;
    error += line_fitter.Fit(&random, half_colus, scan_line, entry_state);
    line_fitter.AppendBestFit(&opcodes_, &states_);
  }
  AppendFrameSuffixOpCodes();
  return error;
}

std::unique_ptr<Image> FrameFitter::SimulateToImage() {
  std::unique_ptr<Image> image(
      new Image(kFrameWidthPixels, kFrameHeightPixels));
  for (uint32 i = 0; i < states_.size(); ++i) {
    states_[i]->PaintInto(image.get());
  }
  return std::move(image);
}

void FrameFitter::SaveBinary(const char* file_name) {
  uint32 total_bytes = 0;
  for (uint32 i = 0; i < opcodes_.size(); ++i)
    total_bytes += opcodes_[i]->bytes();
  std::unique_ptr<uint8[]> bin_buf(new uint8[total_bytes]);
  uint8* buf_ptr = bin_buf.get();
  for (uint32 i = 0; i < opcodes_.size(); ++i)
    buf_ptr += opcodes_[i]->bytecode(buf_ptr);

  int bin_fd = open(file_name, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (bin_fd < 0)
    return;
  write(bin_fd, bin_buf.get(), total_bytes);
  close(bin_fd);
}

void FrameFitter::AppendFramePrefixOpCodes() {
  assert(opcodes_.size() == 0);

  // Turn off VBLANK, turn on VSYNC for three scanlines.
  opcodes_.push_back(makeLDA(0));
  opcodes_.push_back(makeSTA(TIA::VBLANK));
  opcodes_.push_back(makeLDA(2));
  opcodes_.push_back(makeSTA(TIA::VSYNC));
  opcodes_.push_back(makeSTA(TIA::WSYNC));
  opcodes_.push_back(makeSTA(TIA::WSYNC));
  opcodes_.push_back(makeSTA(TIA::WSYNC));

  // Turn off VSYNC, then 37 scanlines of vertical blank, occupy two scanlines
  // with initialization code.
  opcodes_.push_back(makeLDA(0));
  opcodes_.push_back(makeSTA(TIA::VSYNC));
  for (uint32 i = 0; i < kVBlankScanLines - 3; ++i)
    opcodes_.push_back(makeSTA(TIA::WSYNC));

  opcodes_.push_back(makeLDA(0));            // 0
  opcodes_.push_back(makeSTA(TIA::RESP0));   // 2
  opcodes_.push_back(makeSTA(TIA::RESP1));   // 5
  opcodes_.push_back(makeSTA(TIA::NUSIZ0));  // 8
  opcodes_.push_back(makeSTA(TIA::NUSIZ1));  // 11
  opcodes_.push_back(makeSTA(TIA::COLUP0));  // 14
  opcodes_.push_back(makeSTA(TIA::COLUP1));  // 17
  opcodes_.push_back(makeSTA(TIA::COLUPF));  // 20
  opcodes_.push_back(makeSTA(TIA::COLUBK));  // 23
  opcodes_.push_back(makeSTA(TIA::CTRLPF));  // 26
  opcodes_.push_back(makeSTA(TIA::REFP0));   // 29
  opcodes_.push_back(makeSTA(TIA::REFP1));   // 32
  opcodes_.push_back(makeSTA(TIA::PF0));     // 35
  opcodes_.push_back(makeSTA(TIA::PF1));     // 38
  opcodes_.push_back(makeSTA(TIA::PF2));     // 41
  opcodes_.push_back(makeSTA(TIA::AUDC0));   // 44
  opcodes_.push_back(makeSTA(TIA::AUDC1));   // 47
  opcodes_.push_back(makeSTA(TIA::AUDF0));   // 50
  opcodes_.push_back(makeSTA(TIA::AUDF1));   // 53
  opcodes_.push_back(makeSTA(TIA::AUDV0));   // 56
  opcodes_.push_back(makeSTA(TIA::AUDV1));   // 59
  opcodes_.push_back(makeSTA(TIA::GRP0));    // 62
  opcodes_.push_back(makeSTA(TIA::GRP1));    // 65
  opcodes_.push_back(makeSTA(TIA::ENAM0));   // 68
  opcodes_.push_back(makeSTA(TIA::ENAM1));   // 71
  opcodes_.push_back(makeNOP());             // 74

  opcodes_.push_back(makeSTA(TIA::ENABL));   // 0
  opcodes_.push_back(makeSTA(TIA::VDELP0));  // 3
  opcodes_.push_back(makeSTA(TIA::VDELP1));  // 6
  opcodes_.push_back(makeSTA(TIA::VDELBL));  // 9
  opcodes_.push_back(makeSTA(TIA::RESMP0));  // 12
  opcodes_.push_back(makeSTA(TIA::RESMP1));  // 15
  opcodes_.push_back(makeSTA(TIA::HMCLR));   // 18
  opcodes_.push_back(makeLDX(0));            // 20
  opcodes_.push_back(makeLDY(0));            // 22
  opcodes_.push_back(makeSTA(TIA::WSYNC));   // 24

  // Now compute states, to give the optimization code something to start
  // working from.
  states_.reserve(opcodes_.size() + 2);
  states_.push_back(std::unique_ptr<State>(new State()));
  for (uint32 i = 0; i < opcodes_.size(); ++i) {
    states_.push_back(opcodes_[i]->Transform(states_.rbegin()->get()));
  }
}

void FrameFitter::AppendFrameSuffixOpCodes() {
  // Turn on VBLANK, then 30 scanlines of overscan.
  opcodes_.push_back(makeLDA(2));
  opcodes_.push_back(makeSTA(TIA::VBLANK));
  for (uint32 i = 0; i < kOverscanScanLines; ++i) {
    opcodes_.push_back(makeSTA(TIA::WSYNC));
  }
}

}  // namespace vcsmc
