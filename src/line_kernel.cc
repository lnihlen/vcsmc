#include "line_kernel.h"

#include "color_distances_table.h"
#include "opcode.h"
#include "random.h"
#include "range.h"
#include "state.h"

namespace vcsmc {

LineKernel::LineKernel()
      : total_cycles_(0),
        total_bytes_(0),
        sim_error_(std::numeric_limits<float>::max()),
        victories_(0) {
}

LineKernel::~LineKernel() {
}

// We deliberately do not copy the simulation results of this kernel, if any,
// because they are likely to be invalidated by any subsequent mutations to
// the cloned kernel.
std::unique_ptr<LineKernel> LineKernel::Clone() {
  std::unique_ptr<LineKernel> lk(new LineKernel());
  for (uint32 i = 0; i < opcodes_.size(); ++i)
    lk->opcodes_.push_back(opcodes_[i]->Clone());
  lk->total_cycles_ = total_cycles_;
  lk->total_bytes_ = total_bytes_;
  return std::move(lk);
}

// Assumed called on an empty LineKernel, generates a set of load/store pairs
// sufficient to fill one line of CPU time.
void LineKernel::Randomize(Random* random) {
  assert(opcodes_.size() == 0);

  while (total_cycles_ < kScanLineWidthCycles - 5) {
    std::unique_ptr<op::OpCode> op = MakeRandomOpCode(random);
    assert(op);
    total_cycles_ += op->cycles();
    total_bytes_ += op->bytes();
    opcodes_.push_back(std::move(op));
  }

  // Keep twitching about until we are at an appropriate length.
  while (!IsAcceptableLength())
    MutateChangeOpCode(random);
}

void LineKernel::Mutate(Random* random) {
  // Reset simulation status as we are changing our opcodes around.
  sim_error_ = std::numeric_limits<float>::max();

  switch (random->Next() % 2) {
    case 0:
      MutateSwapOpCodes(random);
      break;

    case 1:
      MutateChangeOpCode(random);
      break;

    default:
      assert(false);
      break;
  }

  while (!IsAcceptableLength())
    MutateChangeOpCode(random);
}

void LineKernel::Simulate(const uint8* half_colus, uint32 scan_line,
    const State* entry_state, uint32 lines_to_score) {
  assert(lines_to_score >= 1);
  assert(states_.size() == 0);

  // Transform |entry_state| through the OpCodes to states, then simulate to
  // pixel buffer to compute total error.
  states_.reserve(opcodes_.size() + 2);
  states_.push_back(entry_state->Clone());

  for (uint32 i = 0; i < opcodes_.size(); ++i)
    states_.push_back(opcodes_[i]->Transform(states_.rbegin()->get()));

  uint8 frame_buffer[kFrameWidthPixels * kFrameHeightPixels];
  std::memset(frame_buffer, 0xff, kFrameWidthPixels * kFrameHeightPixels);
  Range paint_range(scan_line * kScanLineWidthClocks,
      (scan_line + lines_to_score) * kScanLineWidthClocks);
  for (uint32 i = 0; i < states_.size(); ++i)
    states_[i]->ColorInto(frame_buffer, paint_range);

  sim_error_ = 0;
  for (uint32 i = 0; i < lines_to_score; ++i) {
    uint32 line = scan_line + i;
    if (line < (kVSyncScanLines + kVBlankScanLines) ||
        line >= (kVSyncScanLines + kVBlankScanLines + kFrameHeightPixels))
      continue;
    uint32 line_offset =
        (line - (kVSyncScanLines + kVBlankScanLines)) * kFrameWidthPixels;
    const uint8* half_colus_line = half_colus + line_offset;
    const uint8* frame_buffer_line = frame_buffer + line_offset;
    for (uint32 j = 0; j < kFrameWidthPixels; ++j) {
      // Ensure all simulated pixels are marked as painted.
      assert(*frame_buffer_line < 128);
      sim_error_ += kAtariColorDistances[
          ((*half_colus_line) * kNTSCColors) + (*frame_buffer_line)];
      ++half_colus_line;
      ++frame_buffer_line;
    }
  }
}

void LineKernel::Compete(LineKernel* lk) {
  assert(IsAcceptableLength());
  if (lk->sim_error_ < sim_error_) {
    ++lk->victories_;
  } else {
    ++victories_;
  }
}

void LineKernel::Append(std::vector<std::unique_ptr<op::OpCode>>* opcodes,
                        std::vector<std::unique_ptr<State>>* states) {
  for (uint32 i = 0; i < opcodes_.size(); ++i)
    opcodes->push_back(opcodes_[i]->Clone());

  // Note that we skip the copy of the entry state we made.
  for (uint32 i = 1; i < states_.size(); ++i)
    states->push_back(states_[i]->Clone());
}

void LineKernel::ResetVictories() {
  victories_ = 0;
}

TIA LineKernel::PickRandomAddress(Random* random) {
  TIA addresses[] = {
//    NUSIZ0,
//    NUSIZ1,
    COLUP0,
    COLUP1,
    COLUPF,
    COLUBK,
    CTRLPF,
    PF0,
    PF1,
    PF2,
    RESP0,
    RESP1,
    GRP0,
    GRP1
  };
  return addresses[random->Next() % 12];
}

std::unique_ptr<op::OpCode> LineKernel::MakeRandomOpCode(Random* random) {
  switch (random->Next() % 7) {
    case 0:
      return makeLDA(random->Next());

    case 1:
      return makeLDX(random->Next());

    case 2:
      return makeLDY(random->Next());

    case 3:
      return makeSTA(PickRandomAddress(random));

    case 4:
      return makeSTX(PickRandomAddress(random));

    case 5:
      return makeSTY(PickRandomAddress(random));

    case 6:
      return makeNOP();

    default:
      assert(false);
      return nullptr;
  }
}

void LineKernel::MutateSwapOpCodes(Random* random) {
  uint32 first_index = random->Next() % opcodes_.size();
  uint32 second_index = random->Next() % opcodes_.size();
  if (first_index == second_index)
    second_index = (second_index + 1) % opcodes_.size();
  std::iter_swap(opcodes_.begin() + first_index,
      opcodes_.begin() + second_index);
}

void LineKernel::MutateChangeOpCode(Random* random) {
  uint32 index = random->Next() % opcodes_.size();
  total_cycles_ -= opcodes_[index]->cycles();
  total_bytes_ -= opcodes_[index]->bytes();
  std::unique_ptr<op::OpCode> op = MakeRandomOpCode(random);
  opcodes_[index].swap(op);
  total_cycles_ += opcodes_[index]->cycles();
  total_bytes_ += opcodes_[index]->bytes();
}

// A scan line occupies 76 CPU cycles. Any kernel exactly 76 cycles is fine.
// Any kernel 73 cycles or shorter is also fine, because we can merely
// terminate the line with a sta WSYNC, which causes the CPU to sleep to the
// end of the scan line. A kernel 74 cycles long is OK, because we can add a
// NOP to the end of the kernel to get it to 76. But a 75 cycle kernel is not
// OK, and anything longer is also not OK.
bool LineKernel::IsAcceptableLength() const {
  return total_cycles_ == 73 || total_cycles_ <= 71;
}

}  // namespace vcsmc
